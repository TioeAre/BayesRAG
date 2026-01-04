def add_vector_store_kg(self, pdf_path):
    from src.langchain.document_parse.mineru import MinerULoader

    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return

    nest_asyncio.apply()

    async def _async_processing_workflow():
        if self.kg is not None:
            await self.kg.rag._ensure_lightrag_initialized()
        text_semaphore = asyncio.Semaphore(1)
        image_semaphore = asyncio.Semaphore(1)
        text_vec_semaphore = asyncio.Semaphore(1)
        image_vec_semaphore = asyncio.Semaphore(1)

        logger.debug(f"Starting background task for PDF level KG: {pdf_path}")

        all_running_tasks = []
        if self.kg is not None:
            pdf_kg_task = asyncio.create_task(self.kg.add_pdf_kg(pdf_path))
            all_running_tasks.append(pdf_kg_task)

        async def _process_vector_batch(docs, batch_type, semaphore):
            """add to embedding vector store"""
            if not docs:
                return
            async with semaphore:
                try:
                    logger.debug(f"[{batch_type}] Vector Store processing started. Size: {len(docs)}")
                    if batch_type == "IMAGE" or batch_type == "IMAGE_FINAL":
                        self.image_embedding.add_image_to_vectorstore(image_documents=docs)
                    elif batch_type == "TEXT" or batch_type == "TEXT_FINAL":
                        self.text_embedding.add_text_to_vectorstore(documents=docs)
                    logger.debug(f"[{batch_type}] Vector Store processing finished.")
                except Exception as e:
                    logger.error(f"Error in {batch_type} Vector Store task: {e}")

        async def _process_batch_kg(docs, batch_type, semaphore):
            if self.kg is None or not docs:
                return
            async with semaphore:
                try:
                    logger.debug(f"[{batch_type}] Batch processing started. Size: {len(docs)}")
                    await asyncio.gather(*[self.kg.add_documents_kg(doc) for doc in docs])  # type: ignore
                    logger.debug(f"[{batch_type}] Batch processing finished.")
                except Exception as e:
                    logger.error(f"Error in {batch_type} KG task: {e}")
                    return

        async def _process_shortcut(path):
            try:
                logger.debug(f"[SHORTCUT] Adding PDF to shortcut vector store: {path}")
                # 如果 add_pdf_to_vectorstore 是同步且耗时的IO操作，建议使用 asyncio.to_thread
                # await asyncio.to_thread(self.shortcut_embedding.add_pdf_to_vectorstore, path)
                self.shortcut_embedding.add_pdf_to_vectorstore(path)
                logger.debug(f"[SHORTCUT] Finished adding PDF to shortcut vector store.")
            except Exception as e:
                logger.error(f"Error in Shortcut Vector Store task: {e}")

        document_loader = MinerULoader(
            file_path=pdf_path,
            table_enable=False,
            backend=project_config.MINERU_BACKEND,
            server_url=project_config.MINERU_SERVER_URL,
            parse_result_dir=project_config.PARSE_RESULT_DIR,
        )
        doc_iterator = document_loader.lazy_load()
        text_batch = []
        image_batch = []
        total_docs_processed = 0
        previous_text_doucment = None

        while True:
            try:
                document = next(doc_iterator)
                total_docs_processed += 1
                if is_base64(document.page_content):
                    image_batch.append(document)
                    if len(image_batch) >= project_config.ADD_VECTOR_IMAGE_BATCH_SIZE:
                        logger.debug(f"Adding batch of {len(image_batch)} image documents...")
                        # self.image_embedding.add_image_to_vectorstore(image_documents=image_batch)
                        task_vec = asyncio.create_task(_process_vector_batch(image_batch, "IMAGE", image_vec_semaphore))
                        all_running_tasks.append(task_vec)
                        task_kg = asyncio.create_task(_process_batch_kg(image_batch, "IMAGE", image_semaphore))
                        all_running_tasks.append(task_kg)
                        image_batch = []
                else:
                    current_text_doucment = document
                    if previous_text_doucment is None:
                        previous_text_doucment = current_text_doucment
                        continue
                    prev_word_counts = document_loader.count_words(previous_text_doucment.page_content)  # type: ignore
                    if prev_word_counts < project_config.MERGE_DOCUMENT_THRESHOLD:
                        merged_document = document_loader.merge_docs(
                            base_doc=previous_text_doucment, new_doc=current_text_doucment  # type: ignore
                        )
                        if len(merged_document) == 1:
                            previous_text_doucment = merged_document[0]
                        else:
                            # 如果不是同一个文档, 或不在同一页
                            text_batch.append(previous_text_doucment)
                            previous_text_doucment = current_text_doucment
                    else:
                        text_batch.append(previous_text_doucment)
                        previous_text_doucment = current_text_doucment
                    if len(text_batch) >= project_config.ADD_VECTOR_TEXT_BATCH_SIZE:
                        logger.debug(f"Adding batch of {len(text_batch)} text documents...")
                        # self.text_embedding.add_text_to_vectorstore(documents=text_batch)
                        task_vec = asyncio.create_task(_process_vector_batch(text_batch, "TEXT", text_vec_semaphore))
                        all_running_tasks.append(task_vec)
                        task_kg = asyncio.create_task(_process_batch_kg(text_batch, "TEXT", text_semaphore))
                        all_running_tasks.append(task_kg)
                        text_batch = []
            except StopIteration:
                logger.debug("Document stream finished. Processing final batches...")
                # if the rest batch is not empty and < ADD_VECTOR_TEXT_BATCH_SIZE
                if previous_text_doucment:
                    text_batch.append(previous_text_doucment)
                if len(text_batch) > 0:
                    logger.debug(f"Adding final batch of {len(text_batch)} text documents...")
                    # self.text_embedding.add_text_to_vectorstore(documents=text_batch)
                    all_running_tasks.append(
                        asyncio.create_task(_process_vector_batch(text_batch, "TEXT_FINAL", text_vec_semaphore))
                    )
                    all_running_tasks.append(
                        asyncio.create_task(_process_batch_kg(text_batch, "TEXT_FINAL", text_semaphore))
                    )

                if len(image_batch) > 0:
                    logger.debug(f"Adding final batch of {len(image_batch)} image documents...")
                    # self.image_embedding.add_image_to_vectorstore(image_documents=image_batch)
                    all_running_tasks.append(
                        asyncio.create_task(_process_vector_batch(image_batch, "IMAGE_FINAL", image_vec_semaphore))
                    )
                    all_running_tasks.append(
                        asyncio.create_task(_process_batch_kg(image_batch, "IMAGE_FINAL", image_semaphore))
                    )
                break
            except Exception as e:
                logger.error(f"Error while processing documents from {pdf_path}: {e}")
                logger.debug(traceback.format_exc())
                text_batch = []
                image_batch = []
                previous_text_doucment = None
                continue
        logger.debug(f"document_loader.lazy_load() processed {total_docs_processed} documents from {pdf_path}")

        if total_docs_processed > 0:
            # self.shortcut_embedding.add_pdf_to_vectorstore(pdf_path)
            shortcut_task = asyncio.create_task(_process_shortcut(pdf_path))
            all_running_tasks.append(shortcut_task)
            logger.debug(f"document_loader.lazy_load() add to vector store {pdf_path}")

        else:
            logger.warning(f"No documents processed from {pdf_path}. Skipping vector store add.")

        if all_running_tasks:
            logger.info(f"Waiting for {len(all_running_tasks)} background (Vector & KG) tasks to complete...")
            await asyncio.gather(*all_running_tasks)
            logger.info("All tasks completed.")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_processing_workflow())
