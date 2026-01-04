import os, sys, copy, base64, asyncio, nest_asyncio, pickle
from loguru import logger


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config

nest_asyncio.apply()


def save_base64_image(base64_string: str, output_path: str):
    """
    Decodes a Base64 string and saves it as an image file.

    Args:
        base64_string: base64 encoded string of the image.
        output_path: the full path to be saved.
    """
    try:
        image_data = base64.b64decode(base64_string)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "wb") as image_file:
            image_file.write(image_data)

        print(f"Image successfully saved to: {output_path}")

    except base64.binascii.Error as e:  # type: ignore
        print(f"Error decoding Base64 string: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def _run_sync(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)


def save_pkl_result(result: dict, pkl_output_path: str):
    """save pkl to disk

    Parameters
    ----------
    result : dict
        result
    pkl_output_path : str
        .pkl path
    """
    if project_config.WRITE_RESULTS:
        pkl_output_dir = os.path.dirname(pkl_output_path)
        os.makedirs(pkl_output_dir, exist_ok=True)

        qa_id = result["qa"]["id"]
        try:
            with open(pkl_output_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Failed to save middle results for {qa_id}: {e}")

        logger.info(f"Results saved to directory: {pkl_output_path}")


# from src.utils.utils import save_pkl_result
