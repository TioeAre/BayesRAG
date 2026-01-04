import cv2
import base64
import numpy as np
import os, sys
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
from src.utils.unigpt import GPT, extract_content_outside_think


class BlockType:
    IMAGE = "image"
    TABLE = "table"


class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class ImageCaptionTester:
    def __init__(self):
        pass

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
            prompt = "Describe this."

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

        print("正在尝试连接 LLM 生成 Caption...")
        try:
            caption_agent = OpenAI(
                base_url=project_config.QWEN3_VL_BASE_URL,
                api_key=project_config.API_KEY,
                timeout=300,  # 测试时超时设置短一点
            )
            completion = caption_agent.chat.completions.create(
                model=project_config.LLM_MODEL_NAME,  # type: ignore
                messages=messages,  # type: ignore
                max_tokens=4096,
            )
            _, response_content = extract_content_outside_think(str(completion.choices[0].message.content))
            caption = response_content
            print("LLM 生成成功。")

        except Exception as e:
            print(f"Warning: LLM 调用失败或超时 (可能是本地服务未启动)。\n错误信息: {e}")
            print(">>> 切换使用【默认测试 Caption】进行后续绘图测试 <<<")
            caption = "Test Caption: This represents a detailed analysis of the visual elements, showing that the system works even without a live LLM connection."

        return caption

    def draw_caption_in_image(self, caption: str, base64image: str) -> str:
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

        # --- 1. 设置字体参数 (Times New Roman 风格) ---
        # 使用 FONT_HERSHEY_COMPLEX 作为衬线体 (Serif) 的替代，接近 Times New Roman 风格
        font = cv2.FONT_HERSHEY_COMPLEX

        # 字体比例设置 (根据图片宽度自适应)
        font_scale = max(0.4, w / 2500.0)
        # 衬线体通常比无衬线体细一点，thickness 稍微设小一点点看起来更像印刷体
        thickness = max(1, int(font_scale * 1.5))

        # 定义边距 (左右留白)
        side_margin = int(30 * font_scale)  # 左对齐时稍微多留点白会更好看

        # 最大允许文本宽度
        max_text_width = w - (side_margin * 2)

        # --- 2. 文本自动换行处理 ---
        words = caption.split(" ")
        lines = []
        current_line = []

        # 获取单行高度信息
        (_, single_line_height), baseline = cv2.getTextSize("Test", font, font_scale, thickness)
        # 行间距设为行高的 1.5 倍左右，符合 Times New Roman 的排版习惯
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

        # --- 3. 计算底部扩展区域高度 ---
        top_padding = int(20 * font_scale)
        bottom_padding = int(20 * font_scale)

        # 计算总扩展高度
        # 注意：最后一行文字的底部还需要一点 padding
        extra_height = top_padding + (len(lines) * line_height) - (line_height - single_line_height) + bottom_padding

        # --- 4. 扩展图像边界 ---
        img_expanded = cv2.copyMakeBorder(img, 0, extra_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # --- 5. 逐行绘制文本 (左对齐) ---
        current_y = h + top_padding + single_line_height

        for line in lines:
            # 左对齐：x 坐标固定为 side_margin
            text_x = side_margin

            cv2.putText(img_expanded, line, (text_x, current_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            current_y += line_height

        # --- 6. 编码返回 ---
        _, buffer = cv2.imencode(".jpg", img_expanded, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")  # type: ignore

        return jpg_as_text


def run_test(local_image_path, output_path):
    tester = ImageCaptionTester()

    # 1. 读取本地图片并转 Base64
    if not os.path.exists(local_image_path):
        print(f"错误: 找不到文件 {local_image_path}")
        return

    with open(local_image_path, "rb") as image_file:
        raw_bytes = image_file.read()
        base64_str = base64.b64encode(raw_bytes).decode("utf-8")

    print(f"1. 图片读取成功: {local_image_path}")

    # 2. 构建 Document 对象
    doc = Document(page_content=base64_str, metadata={"para_type": BlockType.IMAGE})  # 传入不带前缀的 base64

    # 3. 生成 Caption (如果 LLM 不通，会使用默认文本)
    print("2. 正在生成/获取 Caption...")
    generated_caption = tester.generate_caption(doc)
    print(f"   Caption内容: {generated_caption}")

    # 4. 将 Caption 绘制回图片
    print("3. 正在绘制 Caption 到图像底部...")
    # 注意：draw_caption_in_image 返回的是纯 base64 字符串 (根据你提供的代码)
    result_base64 = tester.draw_caption_in_image(generated_caption, base64_str)

    # 5. 保存结果到本地
    try:
        # 处理可能的 data:image 前缀 (虽然你的函数只返回了 base64 字符串)
        if "," in result_base64:
            result_data = result_base64.split(",")[1]
        else:
            result_data = result_base64

        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result_data))
        print(f"4. 成功! 结果已保存至: {output_path}")

    except Exception as e:
        print(f"保存失败: {e}")


if __name__ == "__main__":
    # --- 配置输入输出 ---
    input_img = "./storge/mineru/2303.08559v2/auto/images/0e696086fabb77391771bccfb19ea0a6eb2a5bfc7975b2121811b8e2b9909ff6.jpg"  # 请确保当前目录下有这张图，或者修改为绝对路径
    output_img = "./tmp/test_result.jpg"

    # 为了演示，如果本地没有 input 图片，我们先生成一张纯色图
    if not os.path.exists(input_img):
        print("未找到输入图片，正在生成一张测试图片...")
        dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
        dummy_img[:] = (200, 100, 50)  # 蓝色背景
        cv2.imwrite(input_img, dummy_img)

    run_test(input_img, output_img)
