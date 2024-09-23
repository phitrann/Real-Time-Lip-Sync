import os
from dataclasses import dataclass


@dataclass
class WebConfigs:
    """
    项目所有的配置
    """

    # ==================================================================
    #                             服务器文件配置
    # ==================================================================
    SERVER_FILE_ROOT = r"./results"

    # 商品文件
    PRODUCT_FILE_DIR = "product_files"
    INSTRUCTIONS_DIR = "instructions"
    IMAGES_DIR = "images"

    # 数字人文件
    STREAMER_FILE_DIR = "avatars"
    STREAMER_INFO_FILES_DIR = "streamer_info_files"

    # ==================================================================
    #                             配置文件路径
    # ==================================================================
    STREAMING_ROOM_CONFIG_PATH = r"./configs/streaming_room_cfg.yaml"  # 直播间信息
    CONVERSATION_MESSAGE_STORE_CONFIG_PATH = r"./configs/conversation_message_store.yaml"  # 对话信息

    CONVERSATION_CFG_YAML_PATH: str = r"./configs/conversation_cfg.yaml"  # 微调数据集生成配置

    # ==================================================================
    #                               组件配置
    # ==================================================================
    ENABLE_RAG: bool = True  # True 启用 RAG 检索增强，False 不启用
    ENABLE_TTS: bool = True  # True 启动 tts，False 不启用
    ENABLE_DIGITAL_HUMAN: bool = True  # True 启动 数字人，False 不启用
    ENABLE_AGENT: bool = os.environ.get("ENABLE_AGENT", "true") == "true"  # True 启动 Agent，False 不启用
    ENABLE_ASR: bool = os.environ.get("ENABLE_ASR", "true") == "true"  # True 启动 语音转文字，False 不启用

    # ==================================================================
    #                             数字人 配置
    # ==================================================================

    DIGITAL_HUMAN_GEN_PATH: str = r"./results/avatars"
    DIGITAL_HUMAN_MODEL_DIR: str = r"./models/"
    DIGITAL_HUMAN_BBOX_SHIFT: int = 0
    # DIGITAL_HUMAN_VIDEO_PATH: str = rf"{SERVER_FILE_ROOT}/{STREAMER_FILE_DIR}/{STREAMER_INFO_FILES_DIR}/lelemiao.mp4"
    # DIGITAL_HUMAN_VIDEO_OUTPUT_PATH: str = rf"{SERVER_FILE_ROOT}/{STREAMER_FILE_DIR}/vid_output"
    
    DIGITAL_HUMAN_VIDEO_PATH: str = rf"./data/video/elon.mp4"
    DIGITAL_HUMAN_VIDEO_OUTPUT_PATH: str = rf"./results/avatars/vid_output"

    DIGITAL_HUMAN_FPS: str = 25


@dataclass
class ApiConfig:
    # ==================================================================
    #                               URL 配置
    # ==================================================================
    API_V1_STR: str = "/api/v1"

    USING_DOCKER_COMPOSE: bool = os.environ.get("USING_DOCKER_COMPOSE", "false") == "true"

    # 路由名字和 compose.yaml 服务名对应
    TTS_ROUTER_NAME: str = "tts" if USING_DOCKER_COMPOSE else "0.0.0.0"
    DIGITAL_ROUTER_NAME: str = "digital_human" if USING_DOCKER_COMPOSE else "0.0.0.0"
    ASR_ROUTER_NAME: str = "asr" if USING_DOCKER_COMPOSE else "0.0.0.0"
    LLM_ROUTER_NAME: str = "llm" if USING_DOCKER_COMPOSE else "0.0.0.0"
    BASE_ROUTER_NAME: str = "base" if USING_DOCKER_COMPOSE else "localhost"

    TTS_URL: str = f"http://{TTS_ROUTER_NAME}:8001/tts"
    ASR_URL: str = f"http://{ASR_ROUTER_NAME}:8003/asr"
    LLM_URL: str = f"http://{LLM_ROUTER_NAME}:23333"

    DIGITAL_HUMAN_URL: str = f"http://{DIGITAL_ROUTER_NAME}:8002/digital_human/gen"
    DIGITAL_HUMAN_CHECK_URL: str = f"http://{DIGITAL_ROUTER_NAME}:8002/digital_human/check"
    DIGITAL_HUMAN_PREPROCESS_URL: str = f"http://{DIGITAL_ROUTER_NAME}:8002/digital_human/preprocess"

    BASE_SERVER_URL: str = f"http://{BASE_ROUTER_NAME}:8000{API_V1_STR}"
    CHAT_URL: str = f"{BASE_SERVER_URL}/streamer-sales/chat"
    UPLOAD_PRODUCT_URL: str = f"{BASE_SERVER_URL}/streamer-sales/upload_product"
    GET_PRODUCT_INFO_URL: str = f"{BASE_SERVER_URL}/streamer-sales/get_product_info"
    GET_SALES_INFO_URL: str = f"{BASE_SERVER_URL}/streamer-sales/get_sales_info"
    PLUGINS_INFO_URL: str = f"{BASE_SERVER_URL}/streamer-sales/plugins_info"

    REQUEST_FILES_URL = f"{BASE_SERVER_URL}/files"


# 实例化
WEB_CONFIGS = WebConfigs()
API_CONFIG = ApiConfig()