# -*- coding: utf-8 -*-
"""个人微信群聊 @ 回复机器人。

读取本机已登录微信的新消息：私聊直接回复，群聊仅在 @ 到自己时回复。
回复由 DeepSeek 生成，再通过当前微信窗口发回对应会话。
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GROUP_PREFIX = re.compile(r"^(wxid_[0-9a-zA-Z_-]+):\s*", re.S)
MENTION = re.compile(r"@([^\s\u2005@<]{1,32})")
SKIP_PRIVATE = {
    "newsapp",
    "weixin",
    "fmessage",
    "floatbottle",
    "medianote",
    "qqmail",
    "qmessage",
    "tmessage",
    "mphelper",
}
SYSTEM_TITLES = {
    "filehelper": "文件传输助手",
}
TITLE_COUNT = re.compile(r"[\s]*[（(]\d+[)）]$")


def log(text: str) -> None:
    line = time.strftime("%H:%M:%S ") + text
    print(line, flush=True)
    try:
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        with open(os.path.join(ROOT, "logs", "bot.log"), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def split_group_body(content: str) -> tuple[str, str]:
    matched = GROUP_PREFIX.match(content or "")
    if not matched:
        return "", content or ""
    return matched.group(1), content[matched.end():]


def mention_names(body: str) -> list[str]:
    return MENTION.findall(body or "")


def is_mentioned(body: str, names: set[str]) -> bool:
    hits = [name for name in mention_names(body) if name != "所有人"]
    for hit in hits:
        for mine in names:
            if hit == mine:
                return True
    return False


def strip_mentions(body: str, names: set[str]) -> str:
    text = body or ""
    for name in names:
        text = text.replace("@" + name, " ")
    text = text.replace("\u2005", " ")
    return re.sub(r"\s+", " ", text).strip()


class DeepSeek:
    def __init__(self, api_key: str, model: str, system_prompt: str, max_tokens: int) -> None:
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def ask(self, history: list[dict], question: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": 0.7,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=payload,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return (text or "").strip()


class Bot:
    def __init__(self) -> None:
        load_env(os.path.join(ROOT, ".env"))
        self.config = load_config()
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("缺少 DEEPSEEK_API_KEY，请写入 .env")
        self.llm = DeepSeek(
            api_key,
            self.config.get("model") or "deepseek-chat",
            self.config.get("system_prompt") or "你是微信助手。",
            int(self.config.get("max_tokens") or 800),
        )
        self.history_turns = int(self.config.get("history_turns") or 6)
        self.reply_private = bool(self.config.get("reply_private", True))
        self.reply_all_groups = bool(self.config.get("reply_all_groups", True))
        self.group_whitelist = [str(name) for name in self.config.get("group_whitelist") or []]
        self.self_names = {str(name) for name in self.config.get("self_names") or [] if str(name).strip()}
        self.history: dict[str, list[dict]] = {}
        self.history_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.outgoing: dict[str, deque] = {}
        self.db = None
        self.gui = None
        self.self_wxid = ""

    def start(self) -> None:
        from wechatauto.db import Listener, WeChatDB
        from wechatauto.guia import WeChatGUI

        log("正在连接已登录的微信...")
        account = (self.config.get("account") or "").strip() or None
        self.db = WeChatDB(account=account)
        info = self.db.get_self_info()
        self.self_wxid = info.get("username") or ""
        nick = (info.get("nick_name") or "").strip()
        if nick:
            self.self_names.add(nick)
        log("当前账号：%s" % (nick or self.self_wxid))
        log("群里 @ 这些名字会触发回复：%s" % "、".join(sorted(self.self_names)))
        if self.group_whitelist:
            log("群白名单：%s" % "、".join(self.group_whitelist))
        elif self.reply_all_groups:
            log("群白名单：全部群，仅被 @ 时回复")
        else:
            log("未开启群回复")

        self.gui = WeChatGUI()
        listener = Listener(self.db, interval=1.0)
        listener.add_all(self.on_message, discover=True)
        listener.start()
        log("开始监听。保持微信窗口在前台，不要锁屏。Ctrl+C 停止。")
        self._ping_filehelper()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log("正在停止")
        finally:
            listener.stop()

    def _ping_filehelper(self) -> None:
        try:
            result = self._send("filehelper", "机器人已启动。私聊我，或在群里 @我，我会用 DeepSeek 回复。")
            log("启动通知：%s" % result)
        except Exception as exc:
            log("启动通知发送失败：%s" % exc)

    def on_message(self, msg: dict, _listener) -> None:
        try:
            self._handle(msg)
        except Exception as exc:
            log("处理消息失败：%s" % exc)

    def _handle(self, msg: dict) -> None:
        if msg.get("type") != "文本":
            return
        chat_id = str(msg.get("username") or "")
        content = msg.get("content") or ""
        if not isinstance(content, str) or not chat_id:
            return

        sender_wxid, body = split_group_body(content)
        if self._from_self(msg, sender_wxid):
            return

        is_group = chat_id.endswith("@chatroom")
        if is_group:
            if not self._group_allowed(chat_id):
                return
            if not is_mentioned(body, self.self_names):
                return
            question = strip_mentions(body, self.self_names)
        else:
            if not self.reply_private or not self._is_person(chat_id):
                return
            question = strip_mentions(body, self.self_names)

        if not question:
            question = "你好"
        chat_name = self._display_name(chat_id)
        sender_name = ""
        if sender_wxid:
            sender_name = self.db.get_nickname(sender_wxid) or ""
        log("收到 %s / %s ：%s" % (chat_name or chat_id, sender_name or "私聊", question[:80]))

        try:
            answer = self._complete(chat_id, question)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:180]
            log("DeepSeek HTTP %s %s" % (exc.code, detail))
            answer = "模型接口暂时不可用，请稍后再试。"
        except Exception as exc:
            log("DeepSeek 调用失败：%s" % exc)
            answer = "模型接口暂时不可用，请稍后再试。"

        if not answer:
            return
        if len(answer) > 1500:
            answer = answer[:1500] + "…"
        if is_group and sender_name:
            answer = "@%s %s" % (sender_name, answer)
        result = self._send(chat_id, answer, chat_name)
        log("已回复 %s ：%s" % (chat_name, result))

    def _from_self(self, msg: dict, sender_wxid: str) -> bool:
        is_self = msg.get("sender_id") == 2
        if sender_wxid and sender_wxid == self.self_wxid:
            is_self = True
        sender_username = msg.get("sender_username") or ""
        if self.self_wxid and sender_username == self.self_wxid:
            is_self = True
        if not is_self:
            return False
        # 文件传输助手里的消息发送者永远是自己。用户从手机发来的要回复，
        # 机器人刚发出去的原文要跳过，避免自己回复自己。
        chat_id = str(msg.get("username") or "")
        if chat_id != "filehelper":
            return True
        _, body = split_group_body(msg.get("content") or "")
        return self._is_outgoing(chat_id, body)

    def _mark_outgoing(self, chat_id: str, text: str) -> None:
        bucket = self.outgoing.setdefault(chat_id, deque(maxlen=30))
        bucket.append((text or "").strip())

    def _is_outgoing(self, chat_id: str, text: str) -> bool:
        return (text or "").strip() in self.outgoing.get(chat_id, ())

    def _is_person(self, chat_id: str) -> bool:
        if chat_id in SKIP_PRIVATE or chat_id.startswith("gh_"):
            return False
        return chat_id == "filehelper" or chat_id.startswith("wxid_")

    def _group_allowed(self, chat_id: str) -> bool:
        if self.group_whitelist:
            name = self.db.get_nickname(chat_id) or ""
            return any(item == name or item == chat_id for item in self.group_whitelist)
        return self.reply_all_groups

    def _complete(self, chat_id: str, question: str) -> str:
        with self.history_lock:
            prior = list(self.history.get(chat_id, []))
        answer = self.llm.ask(prior, question)
        with self.history_lock:
            bucket = self.history.setdefault(chat_id, [])
            bucket.append({"role": "user", "content": question})
            bucket.append({"role": "assistant", "content": answer})
            keep = self.history_turns * 2
            if len(bucket) > keep:
                del bucket[:-keep]
        return answer

    def _display_name(self, chat_id: str) -> str:
        """会话在微信窗口标题上的名字。群号本身不能拿去搜索。"""
        name = ""
        try:
            name = (self.db.get_nickname(chat_id) or "").strip()
        except Exception:
            name = ""
        if chat_id.endswith("@chatroom"):
            try:
                alt = (self.db.group_id_to_name(chat_id) or "").strip()
            except Exception:
                alt = ""
            if alt and not alt.endswith("@chatroom"):
                name = alt
        if not self._usable_title(name, chat_id):
            name = SYSTEM_TITLES.get(chat_id, "")
        return name if self._usable_title(name, chat_id) else ""

    @staticmethod
    def _norm_title(name: str) -> str:
        return TITLE_COUNT.sub("", (name or "").strip())

    def _usable_title(self, name: str, chat_id: str) -> bool:
        title = (name or "").strip()
        if not title or title == chat_id:
            return False
        if title.endswith("@chatroom") or title.startswith("wxid_"):
            return False
        return True

    def _same_chat(self, opened: str, expected: str) -> bool:
        left = self._norm_title(opened)
        right = self._norm_title(expected)
        return bool(left) and left == right

    def _send(self, chat_id: str, text: str, chat_name: str | None = None) -> str:
        self._mark_outgoing(chat_id, text)
        target = chat_name or self._display_name(chat_id)
        if not self._usable_title(target, chat_id):
            return "失败 没有群名，已取消发送，避免发到当前打开的聊天"
        try:
            uia = self.gui._get_uia()
        except Exception:
            uia = None
        with self.send_lock:
            if uia is not None and self._same_chat(uia.current_chat(), target):
                if uia.send_text(text):
                    return "成功 已发送到「%s」" % target
            # 只按群名发送。群号搜索会点中结果列表第一项，也就是置顶会话。
            result = self.gui.send_msg(text, target, verify=False)
        status = result.get("status") if isinstance(result, dict) else str(result)
        message = result.get("message") if isinstance(result, dict) else ""
        return "%s %s" % (status, message)


def main() -> None:
    Bot().start()


if __name__ == "__main__":
    main()
