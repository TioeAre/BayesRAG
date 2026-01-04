def add_to_combination_results(
    text: list, image: list, shortcuts: list, A_score: float, BA_score: float, AB_score: float, all_combinations: list
):
    all_combinations.append(
        {
            "text_results": text,
            "image_results": image,
            "shortcut_results": shortcuts,
            "A_score": A_score,
            "BA_score": BA_score,
            "AB_score": AB_score,
        }
    )


def add_to_final_results(result: dict, prefix: str, document_id: str, page_idx: str, final_results: dict):
    if document_id not in final_results.keys():
        final_results[document_id] = {}
    if page_idx not in final_results[document_id].keys():
        final_results[document_id][page_idx] = {
            "text_results": [],
            "image_results": [],
            "shortcut_results": [],
            # "page_score": 0.0,
        }
    if prefix == "shortcut_results" and len(final_results[document_id][page_idx]["shortcut_results"]) == 0:
        # final_results[document_id][page_idx]["shortcut_results"] = merged_results[document_id][page_idx].get(
        #     "shortcut_results", []
        # )
        final_results[document_id][page_idx][prefix].append(result)
    elif prefix != "shortcut_results":
        para_idxs = []
        for final_prefix_result in final_results[document_id][page_idx][prefix]:
            para_idxs.append(final_prefix_result["result"].metadata["para_idx"])
        if result["result"].metadata["para_idx"] not in para_idxs:
            final_results[document_id][page_idx][prefix].append(result)
