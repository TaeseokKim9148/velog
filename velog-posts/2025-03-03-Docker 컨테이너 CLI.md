---
title: "Docker 컨테이너 CLI"
date: Mon, 03 Mar 2025 13:18:31 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Docker-%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88-CLI
---

<h2 id="격리-기술">격리 기술</h2>
<ul>
<li>chroot : 프로세스의 루트 디렉토리를 변경, 격상하여 가상의 루트 디렉터리를 생성</li>
<li>Pivot_root : 루트 파일시스템 자체를 바꿔 컨테이너가 전용 루트 파일 시스템을 가지도록 함</li>
<li>Mount namespace  :  파일 시스템 트리 구성</li>
<li>UTS namespace :  hostname 격리를 수행하여 고유한 hostname 보유</li>
<li>PID namespace : PID와 프로세스 분리</li>
<li>Network namespace : 네트워크 리소스 할당</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/c6b21a12-aaa4-421b-9971-f1961d6871af/image.png" /></p>
<p>docker kill</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/9af3d610-111a-4565-826a-dc02fc24292d/image.png" /></p>
<p>exec/attcach : 작업을 수행할때 / 실제로 돌아가는 상황 확인등</p>
<p>docker diff  : 실행중인 변경상항 확인</p>
<p>docker commit : 실행중인 컨테이너의 변경사항 포함한 새로운 이미지 생성</p>
<p>docker export : 파일로 내보내는것</p>
