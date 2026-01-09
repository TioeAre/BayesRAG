# Copyright (c) Opendatalab. All rights reserved.
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
from src.utils.gpt import GPT, extract_content_outside_think

# isort:skip
import copy
import json
import cv2
import numpy as np
import base64
from pathlib import Path
from loguru import logger
from typing import List
from openai import OpenAI

if project_config.ADD_VECTOR:
    from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
    from mineru.utils.enum_class import BlockType, ContentType, MakeMode

    if project_config.MINERU_BACKEND.startswith("pipeline"):
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.backend.pipeline.model_json_to_middle_json import (
            result_to_middle_json as pipeline_result_to_middle_json,
        )
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text, get_title_level
    from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
    from typing import AsyncIterator, Iterator

    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
    import uuid

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.image import is_base64, image2base64
    from utils.uuid import generate_stable_uuid_for_text

    def do_parse(
        output_dir,  # Output directory for storing parsing results
        pdf_file_names: list[str],  # List of PDF file names to be parsed
        pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
        p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
        backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
        parse_method="auto",  # The method for parsing PDF, default is 'auto'
        p_formula_enable=True,  # Enable formula parsing
        p_table_enable=False,  # INFO Enable table parsing
        server_url=None,  # Server URL for vlm-sglang-client backend
        f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
        f_draw_span_bbox=True,  # Whether to draw span bounding boxes
        f_dump_md=True,  # Whether to dump markdown files
        f_dump_middle_json=True,  # Whether to dump middle JSON files
        f_dump_model_output=True,  # Whether to dump model output files
        f_dump_orig_pdf=True,  # Whether to dump original PDF files
        f_dump_content_list=True,  # Whether to dump content list files
        f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
    ):
        failed_paths = []
        if backend == "pipeline":
            logger.info(f"start to convert_pdf_bytes_to_bytes_by_pypdfium2()")
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                try:
                    new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                        pdf_bytes=pdf_bytes, start_page_id=start_page_id, end_page_id=end_page_id
                    )
                    pdf_bytes_list[idx] = new_pdf_bytes
                except Exception as e:
                    logger.exception(e)
                    failed_paths.append(pdf_file_names[idx])
                    continue

            logger.info(f"read pdf ok! start pipeline_doc_analyze()")
            try:
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    pdf_bytes_list=pdf_bytes_list,
                    lang_list=p_lang_list,
                    parse_method=parse_method,
                    formula_enable=p_formula_enable,
                    table_enable=p_table_enable,
                )
            except Exception as e:
                logger.exception(e)
                failed_paths = pdf_file_names
                return failed_paths
            logger.info(f"pipeline_doc_analyze() ok")
            for idx, model_list in enumerate(infer_results):
                try:
                    model_json = copy.deepcopy(model_list)
                    pdf_file_name = pdf_file_names[idx]
                    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
                    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

                    images_list = all_image_lists[idx]
                    pdf_doc = all_pdf_docs[idx]
                    _lang = lang_list[idx]
                    _ocr_enable = ocr_enabled_list[idx]
                    middle_json = pipeline_result_to_middle_json(
                        model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, p_formula_enable
                    )

                    pdf_info = middle_json["pdf_info"]

                    pdf_bytes = pdf_bytes_list[idx]
                    if f_draw_layout_bbox:
                        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

                    if f_draw_span_bbox:
                        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

                    if f_dump_orig_pdf:
                        md_writer.write(
                            f"{pdf_file_name}_origin.pdf",
                            pdf_bytes,
                        )

                    if f_dump_md:
                        image_dir = str(os.path.basename(local_image_dir))
                        md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                        md_writer.write_string(
                            f"{pdf_file_name}.md",
                            md_content_str,  # type: ignore
                        )

                    if f_dump_content_list:
                        image_dir = str(os.path.basename(local_image_dir))
                        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                        md_writer.write_string(
                            f"{pdf_file_name}_content_list.json",
                            json.dumps(content_list, ensure_ascii=False, indent=4),
                        )

                    if f_dump_middle_json:
                        md_writer.write_string(
                            f"{pdf_file_name}_middle.json",
                            json.dumps(middle_json, ensure_ascii=False, indent=4),
                        )

                    if f_dump_model_output:
                        md_writer.write_string(
                            f"{pdf_file_name}_model.json",
                            json.dumps(model_json, ensure_ascii=False, indent=4),
                        )

                    logger.info(f"local output dir is {local_md_dir}")
                except Exception as e:
                    logger.exception(e)
                    failed_paths.append(pdf_file_names[idx])
                    continue
        else:
            if backend.startswith("vlm-"):
                backend = backend[4:]

            f_draw_span_bbox = False
            parse_method = "vlm"
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                try:
                    pdf_file_name = pdf_file_names[idx]
                    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
                    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
                    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
                    middle_json, infer_result = vlm_doc_analyze(
                        pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url
                    )

                    pdf_info = middle_json["pdf_info"]

                    if f_draw_layout_bbox:
                        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

                    if f_draw_span_bbox:
                        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

                    if f_dump_orig_pdf:
                        md_writer.write(
                            f"{pdf_file_name}_origin.pdf",
                            pdf_bytes,
                        )

                    if f_dump_md:
                        image_dir = str(os.path.basename(local_image_dir))
                        md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                        md_writer.write_string(
                            f"{pdf_file_name}.md",
                            md_content_str,  # type: ignore
                        )

                    if f_dump_content_list:
                        image_dir = str(os.path.basename(local_image_dir))
                        content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                        md_writer.write_string(
                            f"{pdf_file_name}_content_list.json",
                            json.dumps(content_list, ensure_ascii=False, indent=4),
                        )

                    if f_dump_middle_json:
                        md_writer.write_string(
                            f"{pdf_file_name}_middle.json",
                            json.dumps(middle_json, ensure_ascii=False, indent=4),
                        )

                    if f_dump_model_output:
                        model_output = ("\n" + "-" * 50 + "\n").join(infer_result)  # type: ignore
                        md_writer.write_string(
                            f"{pdf_file_name}_model_output.txt",
                            model_output,
                        )

                    logger.info(f"local output dir is {local_md_dir}")
                except Exception as e:
                    logger.exception(e)
                    failed_paths.append(pdf_file_names[idx])
                    continue
        return failed_paths

    def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,  # Start page ID for parsing, default is 0
        end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
        enable_table=False,  # if enable table parsing
    ):
        """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        """

        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        failed_paths = []
        batch_size = 10

        for i in range(0, len(path_list), batch_size):
            batch_paths = path_list[i : i + batch_size]

            file_name_list = []
            pdf_bytes_list = []
            lang_list = []

            for path in batch_paths:
                file_stem = Path(path).stem
                expected_output_dir = Path(output_dir) / file_stem
                if expected_output_dir.is_dir():
                    logger.info(f"{file_stem} exist, skip")
                    continue
                file_name_list.append(file_stem)
                pdf_bytes_list.append(read_fn(path))
                lang_list.append(lang)
            if not file_name_list:
                continue

            logger.info(f"start to process {i//batch_size + 1}, with {len(file_name_list)} files...")
            batch_failed_paths = do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                p_table_enable=enable_table,
            )
            if batch_failed_paths:
                failed_paths.extend(batch_failed_paths)
        return failed_paths

    class MinerUParser:
        def __init__(self) -> None:
            pass

    class MinerULoader(BaseLoader):

        def __init__(
            self,
            file_path: str,
            table_enable: bool,
            formula_enable=True,
            lang="ch",
            backend="pipeline",
            method="auto",
            server_url=None,
            start_page_id: int = 0,
            end_page_id=None,
            parse_result_dir=f"{project_config.project_root}/storge/mineru",
        ) -> None:
            """Initialize the loader with a file path.

            Parameters
            ----------
            file_path : str
                The path to the file to load.
            table_enable : bool
                Enable table parsing to markdown format, else keep tables to images
            """
            self.file_path = file_path
            self.table_enable = table_enable
            self.formula_enable = formula_enable
            self.lang = lang
            self.backend = backend
            self.method = method
            self.server_url = server_url
            self.start_page_id = start_page_id
            self.end_page_id = end_page_id
            self.parse_result_dir = parse_result_dir

        def lazy_load(self) -> Iterator[Document]:
            """A lazy loader that reads a file block by block.

            Yields
            ------
            Iterator[Document]
                [Document] defined in LangChain, each Document is a paragraph or a block in the pdf.
            """
            # parse_result_json = os.path.join(
            #     self.parse_result_dir, pdf_file_name, self.method, f"{pdf_file_name}_middle.json"
            # )
            pdf_file_names = [Path(self.file_path).stem]
            pdf_bytes_list = [read_fn(path=self.file_path)]
            lang_list = [self.lang]

            parse_result = {"failed_paths": []}
            result_path_exist = os.path.join(
                self.parse_result_dir,
                Path(self.file_path).stem,
                self.method,
                f"{Path(self.file_path).stem}_middle.json",
            )
            if not os.path.exists(result_path_exist):
                if self.backend == "pipeline":
                    logger.info(f"start to convert_pdf_bytes_to_bytes_by_pypdfium2()")
                    for idx, pdf_bytes in enumerate(pdf_bytes_list):
                        try:
                            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                                pdf_bytes=pdf_bytes, start_page_id=self.start_page_id, end_page_id=self.end_page_id
                            )
                            pdf_bytes_list[idx] = new_pdf_bytes
                        except Exception as e:
                            logger.exception(e)
                            parse_result["failed_paths"].append(pdf_file_names[idx])
                            continue

                    logger.info(f"read pdf ok! start pipeline_doc_analyze()")
                    try:
                        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
                            pipeline_doc_analyze(
                                pdf_bytes_list=pdf_bytes_list,
                                lang_list=lang_list,
                                parse_method=self.method,
                                formula_enable=self.formula_enable,
                                table_enable=self.table_enable,
                            )
                        )
                    except Exception as e:
                        logger.exception(e)
                        parse_result["failed_paths"] = pdf_file_names
                        # return parse_result
                        raise
                    logger.info(f"pipeline_doc_analyze() ok")
                    for idx, model_list in enumerate(infer_results):
                        try:
                            model_json = copy.deepcopy(model_list)
                            # write results to local dir
                            pdf_file_name = pdf_file_names[idx]
                            local_image_dir, local_md_dir = prepare_env(
                                output_dir=self.parse_result_dir, pdf_file_name=pdf_file_name, parse_method=self.method
                            )
                            image_writer, md_writer = FileBasedDataWriter(
                                parent_dir=local_image_dir
                            ), FileBasedDataWriter(parent_dir=local_md_dir)
                            # parse results to json
                            images_list = all_image_lists[idx]
                            pdf_doc = all_pdf_docs[idx]
                            _lang = lang_list[idx]
                            _ocr_enable = ocr_enabled_list[idx]
                            # write images
                            middle_json = pipeline_result_to_middle_json(
                                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, self.formula_enable
                            )

                            pdf_info = middle_json["pdf_info"]  # NOTE contains layout and span info
                            # visualize parse results
                            pdf_bytes = pdf_bytes_list[idx]
                            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
                            draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
                            # write to markdown
                            image_dir = str(os.path.basename(local_image_dir))
                            md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
                            md_writer.write_string(
                                f"{pdf_file_name}.md",
                                str(md_content_str),
                            )
                            md_writer.write_string(
                                f"{pdf_file_name}_middle.json",
                                json.dumps(middle_json, ensure_ascii=False, indent=4),
                            )
                            logger.info(f"local output dir is {local_md_dir}")
                            md_writer.write_string(
                                f"parse_result.json",
                                json.dumps(parse_result, ensure_ascii=False, indent=2),
                            )
                        except Exception as e:
                            logger.exception(e)
                            parse_result["failed_paths"].append(pdf_file_names[idx])
                            continue
                ### vlm 后端
                else:
                    if self.backend.startswith("vlm-"):
                        backend = self.backend[4:]

                    f_draw_span_bbox = False
                    parse_method = "vlm"
                    for idx, pdf_bytes in enumerate(pdf_bytes_list):
                        try:
                            pdf_file_name = pdf_file_names[idx]
                            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                                pdf_bytes, self.start_page_id, self.end_page_id
                            )
                            local_image_dir, local_md_dir = prepare_env(
                                self.parse_result_dir, pdf_file_name, parse_method
                            )
                            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
                                local_md_dir
                            )
                            middle_json, infer_result = vlm_doc_analyze(
                                pdf_bytes, image_writer=image_writer, backend=backend, server_url=self.server_url
                            )

                            pdf_info = middle_json["pdf_info"]

                            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
                            draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

                            md_writer.write(
                                f"{pdf_file_name}_origin.pdf",
                                pdf_bytes,
                            )
                            image_dir = str(os.path.basename(local_image_dir))
                            md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
                            md_writer.write_string(
                                f"{pdf_file_name}.md",
                                md_content_str,  # type: ignore
                            )
                            md_writer.write_string(
                                f"{pdf_file_name}_middle.json",
                                json.dumps(middle_json, ensure_ascii=False, indent=4),
                            )
                            model_output = ("\n" + "-" * 50 + "\n").join(infer_result)  # type: ignore
                            md_writer.write_string(
                                f"{pdf_file_name}_model_output.txt",
                                model_output,
                            )
                            md_writer.write_string(
                                f"parse_result.json",
                                json.dumps(parse_result, ensure_ascii=False, indent=2),
                            )
                            logger.info(f"local output dir is {local_md_dir}")
                        except Exception as e:
                            logger.exception(e)
                            parse_result["failed_paths"].append(pdf_file_names[idx])
                            continue
            ### 已经解析过了
            else:
                with open(result_path_exist, "r") as f:
                    middle_json = json.load(f)
                local_image_dir = os.path.join(self.parse_result_dir, Path(self.file_path).stem, self.method, f"images")
                image_dir = str(os.path.basename(local_image_dir))
            ### 读取解析结果, 转为 document 对象返回
            for page_info in middle_json["pdf_info"]:
                para_blocks = page_info.get("para_blocks")
                if not para_blocks:
                    continue
                page_idx = page_info.get("page_idx")
                page_size = page_info.get("page_size")
                metadata = {
                    "source": Path(self.file_path).name,
                    "local_image_dir": local_image_dir,
                    # "para_bbox": json.dumps(para_bbox),
                    "page_idx": page_idx,
                    # "para_idx": para_number,
                    "page_size": json.dumps(page_size),
                    "is_image": False,
                    "caption": "",  # only for image and table image
                    # "para_type": para_type,
                    # "uuid": str(uuid.uuid4()),
                    # "image_uuid": "", # str or list[str]
                }
                for i in range(len(para_blocks)):
                    para_block = para_blocks[i]
                    para_number = i
                    documents = self.parse_para(para_block, image_dir, para_number, metadata)
                    self.add_caption_in_image(documents)
                    for document in documents:
                        yield document

        def parse_para(self, para_block: dict, image_dir: str, para_number: int, metadata: dict) -> list[Document]:
            """_summary_

            Parameters
            ----------
            para_block : dict
                paras_of_layout = page_info.get('para_blocks'), is the paragraph in each page
            image_dir : str
                image prefix used in markdown

            Returns
            -------
            tuple[str, str]
                paragraph text, paragraph type(text, title, equation, image, table)
            """
            para_text = ""
            image_uuid = ""
            documents = []
            para_type = para_block["type"]

            para_bbox = para_block.get("bbox")
            metadata["para_bbox"] = json.dumps(para_bbox)
            metadata["para_idx"] = para_number
            # "para_type": para_type,
            # "uuid": str(uuid.uuid4())
            metadata["para_type"] = para_type

            # get normal text
            if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
                para_text = merge_para_with_text(para_block)

                if para_text != "":
                    # metadata["uuid"] = generate_stable_uuid_for_text(para_text)
                    metadata["uuid"] = generate_stable_uuid_for_text(
                        f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                    )
                    document = Document(
                        page_content=para_text,
                        metadata=metadata,
                    )
                    documents.append(document)
            # get title
            elif para_type == BlockType.TITLE:
                title_level = get_title_level(para_block)
                para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'

                if para_text != "":
                    metadata["uuid"] = generate_stable_uuid_for_text(
                        f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                    )
                    document = Document(
                        page_content=para_text,
                        metadata=metadata,
                    )
                    documents.append(document)
            # get equation
            elif para_type == BlockType.INTERLINE_EQUATION:
                if len(para_block["lines"]) == 0 or len(para_block["lines"][0]["spans"]) == 0:
                    # para_text = ""
                    # metadata["uuid"] = generate_stable_uuid_for_text(
                    #     f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                    # )
                    # metadata["para_type"] = ""
                    # document = Document(
                    #     page_content=para_text,
                    #     metadata=metadata,
                    # )
                    # documents.append(document)
                    return documents

                if para_block["lines"][0]["spans"][0].get("content", ""):  # NOTE: 文本的公式, 应当只会返回一个 document
                    para_text = merge_para_with_text(para_block)
                    if para_text != "":
                        metadata["uuid"] = generate_stable_uuid_for_text(
                            f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                        )
                        metadata["is_image"] = False
                        document = Document(
                            page_content=para_text,
                            metadata=metadata,
                        )
                        documents.append(document)
                else:  # NOTE: 图片的公式, 应当只会返回一个 document, 没有 caption, 图片上加上 xxx.pdf, page xxx 的字
                    # para_text += f"![]({image_dir}/{para_block['lines'][0]['spans'][0]['image_path']})"
                    image_path = f"{metadata['local_image_dir']}/{para_block['lines'][0]['spans'][0]['image_path']}"
                    metadata["uuid"] = generate_stable_uuid_for_text(
                        text=para_block["lines"][0]["spans"][0]["image_path"]
                    )
                    metadata["image_path"] = image_path
                    metadata["is_image"] = True
                    document = Document(
                        page_content=image2base64(image_path),
                        metadata=metadata,
                    )
                    documents.append(document)
            # get image
            elif para_type == BlockType.IMAGE:  # FIX
                # image footnote
                has_image_footnote = any(block["type"] == BlockType.IMAGE_FOOTNOTE for block in para_block["blocks"])
                # concat footnote after image captions and cody
                if has_image_footnote:
                    for block in para_block["blocks"]:  # NOTE: image_body, image format
                        if block["type"] == BlockType.IMAGE_BODY:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    if span["type"] == ContentType.IMAGE:
                                        if span.get("image_path", ""):
                                            # para_text += f"![]({image_dir}/{span['image_path']})"
                                            image_path = f"{metadata['local_image_dir']}/{span['image_path']}"
                                            metadata["image_path"] = image_path
                                            metadata["uuid"] = generate_stable_uuid_for_text(span["image_path"])
                                            image_uuid = metadata["uuid"]
                                            metadata["is_image"] = True
                                            document = Document(
                                                page_content=image2base64(image_path),
                                                metadata=metadata,
                                            )
                                            documents.append(document)
                    for block in para_block["blocks"]:
                        # image_caption
                        if block["type"] == BlockType.IMAGE_CAPTION:
                            para_text += merge_para_with_text(block) + "  \n"
                    for block in para_block["blocks"]:
                        # image_footnote
                        if block["type"] == BlockType.IMAGE_FOOTNOTE:
                            para_text += "  \n" + merge_para_with_text(block)
                    if para_text != "":
                        metadata["para_type"] = "image_text"
                        metadata["uuid"] = generate_stable_uuid_for_text(
                            f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                        )
                        if image_uuid != "":
                            metadata["image_uuid"] = image_uuid
                        metadata["is_image"] = False
                        document = Document(
                            page_content=para_text,
                            metadata=metadata,
                        )
                        documents.append(document)
                else:
                    for block in para_block["blocks"]:  # image_body
                        if block["type"] == BlockType.IMAGE_BODY:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    if span["type"] == ContentType.IMAGE:
                                        if span.get("image_path", ""):
                                            # para_text += f"![]({image_dir}/{span['image_path']})"
                                            image_path = f"{metadata['local_image_dir']}/{span['image_path']}"
                                            metadata["image_path"] = image_path
                                            metadata["uuid"] = generate_stable_uuid_for_text(span["image_path"])
                                            image_uuid = metadata["uuid"]
                                            metadata["is_image"] = True
                                            document = Document(
                                                page_content=image2base64(image_path),
                                                metadata=metadata,
                                            )
                                            documents.append(document)
                    for block in para_block["blocks"]:  # image_caption
                        if block["type"] == BlockType.IMAGE_CAPTION:
                            para_text += "  \n" + merge_para_with_text(block)
                    if para_text != "":  # for image captions
                        metadata["para_type"] = BlockType.IMAGE_CAPTION
                        metadata["uuid"] = generate_stable_uuid_for_text(
                            f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                        )
                        if image_uuid != "":
                            metadata["image_uuid"] = image_uuid
                        metadata["is_image"] = False
                        document = Document(
                            page_content=para_text,
                            metadata=metadata,
                        )
                        documents.append(document)

            # get table
            elif para_type == BlockType.TABLE:
                for block in para_block["blocks"]:  # table_body
                    if block["type"] == BlockType.TABLE_BODY:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.TABLE:
                                    # if processed by table model
                                    if span.get("html", ""):
                                        para_text += f"\n{span['html']}\n"
                                    elif span.get("image_path", ""):
                                        # para_text += f"![]({image_dir}/{span['image_path']})"
                                        image_path = f"{metadata['local_image_dir']}/{span['image_path']}"
                                        metadata["image_path"] = image_path
                                        metadata["uuid"] = generate_stable_uuid_for_text(span["image_path"])
                                        image_uuid = metadata["uuid"]
                                        metadata["is_image"] = True
                                        document = Document(
                                            page_content=image2base64(image_path),
                                            metadata=metadata,
                                        )
                                        documents.append(document)
                if para_text != "":  # for table html text
                    metadata["uuid"] = generate_stable_uuid_for_text(
                        f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                    )
                    metadata["is_image"] = False
                    document = Document(
                        page_content=para_text,
                        metadata=metadata,
                    )
                    documents.append(document)

                para_text = ""  # clean above table html text
                for block in para_block["blocks"]:  # table_caption
                    if block["type"] == BlockType.TABLE_CAPTION:
                        para_text += merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:  # table_footnote
                    if block["type"] == BlockType.TABLE_FOOTNOTE:
                        para_text += "\n" + merge_para_with_text(block) + "  "

                if para_text != "":  # for table captions and footnotes
                    metadata["para_type"] = "table_text"
                    metadata["uuid"] = metadata["uuid"] = generate_stable_uuid_for_text(
                        f"{metadata['source']}_{metadata['page_idx']}_{metadata['para_idx']}_{metadata['para_type']}"
                    )
                    if image_uuid != "":
                        metadata["image_uuid"] = image_uuid
                    metadata["is_image"] = False
                    document = Document(
                        page_content=para_text,
                        metadata=metadata,
                    )
                    documents.append(document)

            return documents

        def count_words(self, text: str) -> int:
            if not text or not isinstance(text, str):
                return 0
            return len(text.split())

        def calculate_bbox_union(self, bbox1, bbox2):
            union_bbox = []
            if len(bbox1) == 4:
                union_bbox = bbox1
            if len(bbox2) == 4:
                union_bbox = bbox2
            if len(bbox1) != 4 or len(bbox2) != 4:
                return union_bbox
            x1a, y1a, x2a, y2a = bbox1
            x1b, y1b, x2b, y2b = bbox2
            union_bbox = [
                min(int(x1a), int(x1b)),
                min(int(y1a), int(y1b)),
                max(int(x2a), int(x2b)),
                max(int(y2a), int(y2b)),
            ]
            return union_bbox

        def merge_docs(self, base_doc: Document, new_doc: Document) -> List[Document]:
            # TODO: merge metadata
            base_meta = base_doc.metadata
            new_meta = new_doc.metadata
            if base_meta.get("source", "") == new_meta.get("source", ""):
                if base_meta.get("page_idx", "") == new_meta.get("page_idx", ""):
                    # 在同一文档的同一页
                    base_doc.page_content += f"\n{new_doc.page_content}"
                    base_bboxes_str = base_meta.get("para_bbox", "[]")
                    loaded_base_bboxes = json.loads(base_bboxes_str)
                    new_bbox_str = new_meta.get("para_bbox", "[]")
                    new_bbox = json.loads(new_bbox_str)
                    unioned_bbox = self.calculate_bbox_union(loaded_base_bboxes, new_bbox)
                    base_meta["para_bbox"] = json.dumps(unioned_bbox)
                    return [base_doc]
            return [base_doc, new_doc]

        def add_caption_in_image(self, documents: List[Document]) -> List[Document]:
            if len(documents) == 0:
                return documents
            if len(documents) != 2:  # text
                if not is_base64(documents[0].page_content):  # no caption image (equation)
                    return documents
            with_caption_docs = []
            caption = ""
            caption_document = None
            image_document = None
            if len(documents) == 2:
                for doc in documents:
                    if is_base64(doc.page_content):
                        image_document = doc
                    else:
                        caption_document = doc
                        caption = caption_document.page_content
            # for only one image, to generate caption that is not equation
            else:
                doc = documents[0]
                if is_base64(doc.page_content):
                    image_document = doc
                    if doc.metadata["para_type"] != BlockType.INTERLINE_EQUATION:
                        if caption == "":
                            caption = self.generate_caption(image_document)

            # draw caption to image_document
            if image_document:
                caption2draw = f"The content is in page {image_document.metadata['page_idx']+1} of {image_document.metadata['source']}. {caption}"
                image_document.page_content = self.draw_caption_in_image(caption2draw, image_document.page_content)
                with_caption_docs.append(image_document)

            if caption_document:
                with_caption_docs.append(caption_document)

            if len(with_caption_docs) != len(documents):
                return documents
            return with_caption_docs

        def generate_caption(self, document: Document) -> str:
            caption = ""
            base64_image = document.page_content
            para_type = document.metadata["para_type"]

            if para_type == BlockType.IMAGE:
                prompt = """Please analyze this image in detail and provide a brief visual caption of the image following these guidelines:
- Identify objects, people, text, and visual elements
- Explain relationships between elements
- Note colors, lighting, and visual style
- Describe any actions or activities shown
- Include technical details if relevant (charts, diagrams, etc.)
- Always use specific names instead of pronouns

Focus on providing accurate, only return a brief visual caption that would be useful for knowledge retrieval and not exceed 100 words."""
            elif para_type == BlockType.TABLE:
                prompt = """Please analyze this table content and provide a brief caption of the table including:
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    Always use specific names and values instead of general references.

    Focus on extracting meaningful insights and relationships from the tabular data, only return a brief visual caption that would be useful for knowledge retrieval and not exceed 100 words."""
            else:
                prompt = "Describe this, only return a brief visual caption that would be useful for knowledge retrieval and not exceed 100 words."

            content = list()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )
            messages = [{"role": "user", "content": content}]
            # caption_agent = GPT(model="azure-gpt-4o", vendor="azure", stream=False, temperature=0.1)
            # response_content, _, _, _, judge_token_usage = caption_agent.send_chat_request(messages=messages)
            caption_agent = OpenAI(
                base_url=project_config.QWEN3_VL_BASE_URL,
                api_key=project_config.LLM_MODEL_API_KEY,
                timeout=3600,
            )
            completion = caption_agent.chat.completions.create(
                model=project_config.LLM_MODEL_NAME,  # type: ignore
                messages=messages,  # type: ignore
                max_tokens=4096,
            )
            _, response_content = extract_content_outside_think(str(completion.choices[0].message.content))

            caption = response_content

            return caption

        def draw_caption_in_image(self, caption: str, base64image: str) -> str:
            # tmp_dir: str = f"{project_config.project_root}/tmp"
            try:
                if "," in base64image:
                    base64_data = base64image.split(",")[1]
                else:
                    base64_data = base64image
                img_bytes = base64.b64decode(base64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    print("Error: Failed to decode image.")
                    return base64image
            except Exception as e:
                print(f"Error processing base64: {e}")
                return base64image

            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_COMPLEX

            # 字体比例设置 (根据图片宽度自适应)
            font_scale = max(0.4, w / 2500.0)
            thickness = max(1, int(font_scale * 1.5))

            side_margin = int(30 * font_scale)
            max_text_width = w - (side_margin * 2)
            words = caption.split(" ")
            lines = []
            current_line = []

            (_, single_line_height), baseline = cv2.getTextSize("Test", font, font_scale, thickness)
            line_height = int((single_line_height + baseline) * 1.5)

            for word in words:
                test_line = " ".join(current_line + [word])
                (text_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_w <= max_text_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []

            if current_line:
                lines.append(" ".join(current_line))

            if not lines:
                lines = [""]
            top_padding = int(20 * font_scale)
            bottom_padding = int(20 * font_scale)

            extra_height = (
                top_padding + (len(lines) * line_height) - (line_height - single_line_height) + bottom_padding
            )
            img_expanded = cv2.copyMakeBorder(img, 0, extra_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            current_y = h + top_padding + single_line_height

            for line in lines:
                text_x = side_margin
                cv2.putText(
                    img_expanded, line, (text_x, current_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
                )

                current_y += line_height

            _, buffer = cv2.imencode(".jpg", img_expanded, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")  # type: ignore

            return jpg_as_text
