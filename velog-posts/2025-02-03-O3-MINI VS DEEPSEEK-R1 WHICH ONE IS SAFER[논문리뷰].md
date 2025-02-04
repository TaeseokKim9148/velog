---
title: "O3-MINI VS DEEPSEEK-R1: WHICH ONE IS SAFER?[논문리뷰]"
date: Mon, 03 Feb 2025 09:12:14 GMT
categories: Velog
link: https://velog.io/@kim_taixi/O3-MINI-VS-DEEPSEEK-R1-WHICH-ONE-IS-SAFER
---

<h3 id="논문-제목o3-mini-vs-deepseek-r1-which-one-is-safer">논문 제목:O3-MINI VS DEEPSEEK-R1: WHICH ONE IS SAFER?</h3>
<h3 id="저자--aitor-arrieta-miriam-ugarte-pablo-valle-josé-antonio-parejo-sergio-segura">저자 : Aitor Arrieta, Miriam Ugarte, Pablo Valle, José Antonio Parejo, Sergio Segura</h3>
<blockquote>
<h2 id="논문-정리">논문 정리</h2>
</blockquote>
<p> LLM은 안전성과 인간의 가치에 대한 정렬(alignment)이라는 중요한 질적 특성을 충족해야한다. 그래서 DeepSeek-R1(70B 버전)과 OpenAI의 o3-mini(베타 버전)의 안전성 수준을 비교하기 위해서 ASTRAL이라는 자동화된 안전성 테스트 도구활용하여 두모델을 대상으로 1,260개의 테스트 입력을 자동화하고 생성하고 실행</p>
<ul>
<li>DeepSeek-R1의 비안전한 응답 비율: 12%<ul>
<li>OpenAI o3-mini의 비안전한 응답 비율: 1.2%</li>
</ul>
</li>
</ul>
<p><strong>DeepSeek-R1이 o3-mini에 비해 10배 더 많은 비안전한 응답을 생성하는 것으로 분석</strong></p>
<p>o3-mini가 높은 안전성을 보인 이유는 강력한 <strong>안전 장치(guardrails)</strong>로 인해 많은 비안전한 프롬프트를 처리하기전에 차단하였으며, 정책위반 메세지를 변환</p>
<h3 id="llm-안전성-테스트-기법">LLM 안전성 테스트 기법</h3>
<ol>
<li>다지선다형 질문 기반 테스트</li>
<li>LLM을 활용한 안전성 검사 모델 개발(LlamaGuard, ShieldLM)</li>
<li>Red Teaming 및 Jailbreak 공격을 통한 테스트<ul>
<li>Red Teaming : 사람이 직접 테스트 입력</li>
<li>Jailbreak : 차단해야 할 정보를 제공하도록 유도 방법</li>
</ul>
</li>
<li>대규모 벤치마크 기반 테스트</li>
</ol>
<hr />
<h4 id="astral">ASTRAL</h4>
<ul>
<li>블랙박스 커버리지 기준(black-box coverage criterion)을 활용하여 비안전한 테스트 입력을 자동 생성</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/23ea1f01-3605-428e-b8c0-3d6d53ee458f/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/3d565778-e163-44f1-a235-980b5554a622/image.png" /></p>
<ul>
<li><p>균형 잡힌(balanced) 최신 프롬프트 생성 가능</p>
</li>
<li><p>RAG(Retrieval-Augmented Generation), few-shot prompting, 웹 검색(web browsing) 전략을 통합하여 테스트 프롬프트를 동적으로 생성</p>
</li>
</ul>
<h3 id="astral의-주요-단계-요약">ASTRAL의 주요 단계 요약</h3>
<p>1️⃣ 테스트 입력 생성 단계 (Test Generation Phase)
LLM이 N개의 비안전한 테스트 입력을 생성
카테고리, 작성 스타일, 설득 기법을 균형 있게 반영하도록 블랙박스 커버리지 기준 적용
OpenAI의 Assistant API를 활용해 RAG 기반 방법을 통합
최신 뉴스 검색 기능을 추가하여 실시간 이슈 반영 가능</p>
<p>2️⃣ 테스트 실행 단계 (Execution Phase)
ASTRAL이 생성된 테스트 입력을 대상 LLM에 입력
LLM이 비안전한 입력을 어떻게 처리하는지 평가</p>
<p>3️⃣ 평가 단계 (Evaluation Phase)
또 다른 LLM이 오라클 역할 수행
대상 LLM의 응답이 안전성 기준을 충족하는지 분석
안전성 여부 최종 판단
<strong>GPT-3.5를 평가 모델로 사용</strong></p>
<p>** 테스트 기준 **</p>
<p>6(작성 스타일)×5(설득 기법)×14(안전성 카테고리)×3(테스트 횟수)=1,260개</p>
<h4 id="테스트-입력-생성-시점">테스트 입력 생성 시점</h4>
<p>ASTRAL은 인터넷에서 정보를 검색하여 최신 데이터를 반영하므로,2024년 미국 대선(US elections)과 관련된 뉴스가 다수 포함됨
특히 C3 카테고리(논란이 있는 주제 및 정치)에서 많은 테스트 입력이 대선 관련 내용을 포함</p>
<h4 id="llm이-생성한-응답을-다음-세-가지로-분류">LLM이 생성한 응답을 다음 세 가지로 분류</h4>
<p>Safe(안전함)
Unsafe(비안전함)
Unknown(판단 불가)</p>
<h3 id="결과">결과</h3>
<p>DeepSeek-R1이 금융 범죄(c6), 폭력(c14), 테러(c13), 혐오 발언(c7) 카테고리에서 높은 위험성을 보임
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/4a59ba87-7a34-4591-a88e-5d4a0b563da7/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/120569c0-c00b-4b27-be4c-a6dbcfcc8494/image.png" /></p>
<p>** 내생각**</p>
<p>평가모델이 GPT-3.5이라는게 조금 불공정한거 같다.</p>
<h2 id="참고자료">참고자료</h2>
