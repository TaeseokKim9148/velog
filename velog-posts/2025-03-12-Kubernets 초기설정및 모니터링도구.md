---
title: "Kubernets 초기설정및 모니터링도구"
date: Wed, 12 Mar 2025 08:18:28 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Kubernets-%EC%B4%88%EA%B8%B0%EC%84%A4%EC%A0%95
---

<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/824bb62a-3366-4a35-9822-fa5a1475de5e/image.png" /></p>
<h2 id="1-설치과정-및-환경설정-완료-">1. 설치과정 및 환경설정 완료 <del>~</del></h2>
<h2 id="관리도구">관리도구</h2>
<ul>
<li>Monitoring : IT시스템에서 GPU사용량 메모리 사용량등 데이터를 수집, 분석해서 동작을 파악하여 시스템에 문제가 있는것으로 추정되는 동작 및 조건을 감지 메트릭이나 로그에 의존</li>
<li>Observability : 관측가능성이란 시스템에서 외부로 출력되는 값만을 사용하여 시스템 내부 상태 예측, 내부 시스템에 대해 이해를 근거로 발생 가능한 이벤트 예측하고 이 예측을 바탕으로 IT운영 자동화</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/a0b91c53-4e8c-485e-9fa2-b8f40573c92b/image.png" /></p>
<h3 id="대시보드-접속">대시보드 접속</h3>
<h3 id="prometheus--grafana">Prometheus &amp; Grafana</h3>
<p>Prometheus는 CLCF에서 제공하는 오픈소스 Metric pipeline을 제공하여 클러스터 및 컨테이너에 대한 편리한 모니터링을 제공</p>
<p>Grafana는 지표를 분석, 시각화 하는 도구로 주로 시각화를 위한 대시보드로 사용</p>
<h3 id="kubeshark--wireshark의-kunernetes버전">kubeshark : Wireshark의 Kunernetes버전</h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/b584b41b-15bd-41c7-a019-284b315dcf2e/image.png" /></p>
<h3 id="portainerio-접속">Portainer.io 접속</h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/35b331ac-264d-456c-b008-0e2ec8b937ad/image.png" /></p>
<h3 id="하나하나-접속과-실행이--쉽지않다">하나하나 접속과 실행이  쉽지않다……</h3>
<h3 id="k9s">K9s</h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/27f4cb03-9821-493d-91ed-61c02a9d7122/image.png" /></p>
