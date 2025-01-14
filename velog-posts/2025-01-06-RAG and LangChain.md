---
title: "RAG and LangChain"
date: Mon, 06 Jan 2025 07:50:18 GMT
categories: Velog
link: https://velog.io/@kim_taixi/RAG-and-LangChain
---

<h1 id="rag">RAG</h1>
<h3 id="정의">정의</h3>
<p>대형 언어 모델(LLMs)<strong>이 외부 컨텍스트를 활용해 **환각(hallucination)</strong> 현상을 줄이고 <strong>정확도</strong>를 높이는 기술</p>
<ul>
<li><strong>Retrieval (검색 단계)</strong><ul>
<li>사용자의 <strong>질문</strong>(query)을 기반으로 외부 지식 소스에서 <strong>추가적인 컨텍스트</strong>를 검색</li>
<li>외부 지식 소스는 여러 정보 조각과 그에 해당하는 <strong>벡터 임베딩</strong>을 저장</li>
<li>검색 시점에 사용자 질문은 동일한 벡터 공간에 임베딩되고, <strong>가장 가까운 데이터 포인트</strong>를 계산하여 유사한 컨텍스트를 검색</li>
</ul>
</li>
<li><strong>Augmentation (증강 단계)</strong><ul>
<li>사용자 질문과 검색된 <strong>추가 컨텍스트</strong>를 사용해 프롬프트 템플릿을 <strong>보강</strong></li>
</ul>
</li>
<li><strong>Generation (생성 단계)</strong><ul>
<li>보강된 프롬프트를 기반으로 <strong>더 사실적이고 정확한 답변</strong>을 생성</li>
<li>이 과정은 단순히 사용자 질문만을 사용했을 때보다 더 높은 정확도를 제공</li>
</ul>
</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/f4de691d-20ba-4bdd-830e-3e9bdcc8092b/image.png" /></p>
<h3 id="retrieval-augmented-generation-forknowledge-intensive-nlp-tasks"><em>Retrieval-Augmented Generation forKnowledge-Intensive NLP Tasks</em></h3>
<ul>
<li>RAG 모델에서 파라메트릭 메모리는 사전 학습된 <strong>seq2seq 모델</strong>이고, 비파라메트릭 메모리는 <strong>사전 학습된 신경 검색기(neural retriever)</strong>를 통해 접근할 수 있는 Wikipedia의 밀집 벡터 인덱스</li>
<li>두 가지 구조를 비교하며, 하나는 전체 시퀀스에 동일한 검색 결과를 사용하는 방식이고, 다른 하나는 토큰별로 다른 검색 결과를 사용하는 방식</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/f2a74f1e-26f2-4b07-a6bd-e9cc16790a2c/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/518cdffe-ba54-4416-9762-a0033cfe8421/image.png" /></p>
<h1 id="langchain">LangChain</h1>
<p>LangChain은 대규모 언어 모델(LLM)을 기반으로 애플리케이션을 구축하기 위한 프레임워크</p>
<ul>
<li>LLM을 기존 데이터 소스(데이터베이스, 문서 등)와 통합하거나, 사용자 정의 워크플로를 구성</li>
<li>LangChain은 주로 RAG와 같은 LLM 기반 기술과 결합하여 정보를 검색하고 생성하는 데 자주 활용</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/c88b5856-2670-41b8-b166-dc191379f1ea/image.webp" /></p>
<h2 id="참고자료">참고자료</h2>
<p><a href="https://www.youtube.com/watch?v=gtOdvAQk6YU">https://www.youtube.com/watch?v=gtOdvAQk6YU</a></p>
