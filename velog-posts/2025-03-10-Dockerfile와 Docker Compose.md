---
title: "Dockerfile와 Docker Compose"
date: Mon, 10 Mar 2025 13:43:26 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Dockerfile%EC%99%80-Docker-Compose
---

<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/95530efc-f7fd-4b23-bf82-764436ea3f15/image.png" /></p>
<ul>
<li>Dockerfile : 원하는 환경의 Docker Image개발에 필요한 instruction을 포함한 텍스트 파일</li>
<li>Dcoker build :  Dockerfile을 사용한 docker image 생성 과정을 트리거하는 Docker CLI</li>
<li>Image registry : 생성된 이미지를 Public or Private하게 저장할 수 있는 영역</li>
<li>Docker image에 포함된 애플리케이션 인프라에 프로세스를 붙여 서비스로 배포되는것 컨테이너라고 함</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/e9618665-b083-4f25-b0d9-12f803ae298b/image.png" /></p>
<h3 id="volume-생성">Volume 생성</h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/6486370b-2fa1-4edf-8fba-5bb1f5cb2fcb/image.png" /></p>
<h3 id="두-컨테이너-생성-및-연결">두 컨테이너 생성 및 연결</h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/24013969-cf74-411e-a7f5-68672d07fd7c/image.png" /></p>
<h3 id="워드프래스설치">워드프래스설치</h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/d62208ee-873a-468d-b411-165a73dfaeb8/image.png" /></p>
<h3 id="yaml-코드-작성">yaml 코드 작성</h3>
<p><a href="https://lejewk.github.io/yaml-syntax/">https://lejewk.github.io/yaml-syntax/</a></p>
