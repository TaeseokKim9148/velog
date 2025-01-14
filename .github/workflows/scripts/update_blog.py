import feedparser
import os
import re
from datetime import datetime
import git
from pathlib import Path

def sanitize_filename(title):
    """파일 이름으로 사용할 수 없는 문자 제거"""
    return re.sub(r'[\\/*?:"<>|]', "", title)

def create_markdown(item):
    """마크다운 파일 내용 생성"""
    content = f"""---
title: "{item.title}"
date: {item.published}
categories: Velog
link: {item.link}
---

{item.description}
"""
    return content

def main():
    # velog-posts 디렉토리 생성
    posts_dir = Path("velog-posts")
    posts_dir.mkdir(exist_ok=True)

    # Velog RSS 피드 가져오기
    feed = feedparser.parse("https://v2.velog.io/rss/kim_taixi")  # 본인의 Velog 아이디로 변경

    # 각 포스트를 마크다운 파일로 저장
    for item in feed.entries:
        # 파일명 생성 (날짜-제목.md)
        date = datetime.strptime(item.published, "%a, %d %b %Y %H:%M:%S %Z").strftime("%Y-%m-%d")
        filename = f"{date}-{sanitize_filename(item.title)}.md"
        file_path = posts_dir / filename

        # 마크다운 파일 생성
        content = create_markdown(item)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Git 커밋
    try:
        repo = git.Repo()
        repo.index.add(["velog-posts/*"])
        repo.index.commit("📝 벨로그 포스트 자동 업데이트")
    except Exception as e:
        print(f"Git 작업 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
