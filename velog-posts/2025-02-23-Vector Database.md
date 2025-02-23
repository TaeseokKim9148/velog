---
title: "Vector Database"
date: Sun, 23 Feb 2025 13:16:24 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Vector-Database
---

<h2 id="vector-db">Vector DB</h2>
<p>Vector 형태의 Embedding을 사용하여 데이터를 저장 인덱싱하는 데이터 베이스
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/48c0f1c9-3c21-48b1-9c17-c4977b912a88/image.png" /></p>
<h3 id="기존-데이터-베이스와-다른점">기존 데이터 베이스와 다른점</h3>
<ul>
<li>빠른 검색 및 대량의 데이터를 처리할 수있는 확장성</li>
<li>복잡한 데이터에 특히 적합한 고차원 백터를 저장하고 처리하느데 중점을 둠</li>
</ul>
<h3 id="종류">종류</h3>
<ul>
<li>Pinecone - API를 통해 사용자가 관리하는 클라우드 기반 벡터 데이터 베이스 </li>
<li>Chroma - 텍스트 문서를 쉽게 처리하고 임베딩으로 변환하고 유사도 검색 수행기능이 있음</li>
<li>FAISS - 유사성 검색 및 고밀도 벡터 클러스터링</li>
<li>Elastic Search - 다양한 유형의 데이터를 지원하는 분산 검색 및 분석 엔진</li>
<li>Weaviate</li>
<li>Qdrant</li>
</ul>
<h2 id="참고자료">참고자료</h2>
<p> <a href="https://discuss.pytorch.kr/t/gn-vector-database/1516">https://discuss.pytorch.kr/t/gn-vector-database/1516</a>
 <a href="https://hotorch.tistory.com/406">https://hotorch.tistory.com/406</a>
 <a href="https://meetcody.ai/ko/blog/2024%EB%85%84%EC%97%90-%EC%8B%9C%EB%8F%84%ED%95%B4-%EB%B3%BC-%EB%A7%8C%ED%95%9C-%EC%83%81%EC%9C%84-5%EA%B0%80%EC%A7%80-%EB%B2%A1%ED%84%B0-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4/">https://meetcody.ai/ko/blog/2024%EB%85%84%EC%97%90-%EC%8B%9C%EB%8F%84%ED%95%B4-%EB%B3%BC-%EB%A7%8C%ED%95%9C-%EC%83%81%EC%9C%84-5%EA%B0%80%EC%A7%80-%EB%B2%A1%ED%84%B0-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4/</a></p>
