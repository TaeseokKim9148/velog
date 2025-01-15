---
title: "Precise Zero-Shot Dense Retrieval without Relevance Labels[논문리뷰]"
date: Wed, 15 Jan 2025 16:11:06 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Precise-Zero-Shot-Dense-Retrieval-without-Relevance-Labels%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0
---

<h3 id="논문-제목precise-zero-shot-dense-retrieval-without-relevance-labels">논문 제목:Precise Zero-Shot Dense Retrieval without Relevance Labels</h3>
<h3 id="저자--luyu-gao-xueguang-majimmy-linjamie-callan">저자 : Luyu Gao, Xueguang Ma,Jimmy Lin,Jamie Callan</h3>
<blockquote>
<h2 id="abstract">Abstract</h2>
</blockquote>
<p>HyDE는 언어 모델(e.g. InstructGPT)을 제로샷으로 활용해 가상의 문서를 생성하고, 비지도 대조 학습 인코더(e.g. Contriever)로 이를 임베딩 벡터로 변환하여 유사한 실제 문서를 검색합니다. 생성된 문서는 관련성 패턴을 포착하지만 실제 문서는 아니며, 인코더의 조밀한 병목 구조가 잘못된 세부 정보를 필터링해 실제 말뭉치와 연결합니다.</p>
<blockquote>
<h2 id="introduction">Introduction</h2>
</blockquote>
<ul>
<li>생성 단계: 언어 모델(e.g. InstructGPT)을 사용해 질의에 기반한 가상의 문서를 생성합니다. 이 문서는 관련성 패턴을 포착하지만, 실제 문서가 아니며 오류가 포함될 수 있음</li>
<li>검색 단계: 생성된 문서를 비지도 대조 학습 인코더(e.g. Contriever)로 임베딩하여, 실제 말뭉치에서 유사한 문서를 검색합니다. 인코더는 가상 문서의 오류나 불필요한 세부 정보를 필터링하며, 문서 간 유사성을 활용해 적합한 결과를 반환</li>
<li>HyDE는 별도의 모델 학습 없이도 작동하며, 웹 검색, 질의응답, 사실 검증과 같은 다양한 작업에서 기존 최첨단 시스템을 능가하는 성능을 보여줌.  특히 한국어, 일본어, 스와힐리어와 같은 여러 언어에서도 강력한 성능을 입증함</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/56b9c984-bf46-4303-9ce6-847a85552fda/image.png" /></p>
<h2 id="참고자료">참고자료</h2>
<p><a href="https://digitalbourgeois.tistory.com/482">https://digitalbourgeois.tistory.com/482</a>
<a href="https://www.koreaodm.com/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5/hyde%EB%A1%9C-rag-%ED%96%A5%EC%83%81%EC%8B%9C%ED%82%A4%EA%B8%B0-%EC%9D%B4%EB%A1%A0%EA%B3%BC-%EC%A0%81%EC%9A%A9-%EB%B0%A9%EC%95%88/">https://www.koreaodm.com/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5/hyde%EB%A1%9C-rag-%ED%96%A5%EC%83%81%EC%8B%9C%ED%82%A4%EA%B8%B0-%EC%9D%B4%EB%A1%A0%EA%B3%BC-%EC%A0%81%EC%9A%A9-%EB%B0%A9%EC%95%88/</a></p>
