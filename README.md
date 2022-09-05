![image](https://user-images.githubusercontent.com/100823210/187594402-4f6a7234-293d-4938-adcb-eb602a50b564.png)
# [ML] 자율주행 센서의 안테나 성능 예측

## 개요
데이콘에서 주최한 대회에 참가하여 딥러닝으로 제품의 성능을 예측하는 프로젝트를 진행했습니다.
라이다 제조 공정에서 발생한 3만 9천개 데이터를 학습 시켰고, 주최측에서 제공한 평가산식(NRMSE) 기준 1.95231 를 얻었습니다.
대회 참가팀 743개 팀 중에 12위로 상위권 성적을 받을 수 있었습니다.

![image](https://user-images.githubusercontent.com/100823210/187594733-4becdacd-6606-4745-bf07-8c2c72e41177.png)

## 참여배경

4차 산업이 발전하면서 AI를 이용한 스마트 공장을 도입하는 중소기업이 늘어나고 있습니다. 제품의 불량율을 예측하는 부분에서 많이 활용되고 있으며, 기업의 경쟁력을 높여주고 있습니다. 이번 프로젝트를 통해 Deep Learning 으로 제조공정의 데이터를 활용하여 불량율을 예측하는 방

> [YTN - 제조 혁신의 아이콘 '스마트 공장'...중소·중견 기업으로 확대](https://n.news.naver.com/mnews/article/052/0001756416?sid=102)

> [ETRI - 자율주행이 불러올 새로운 이동수단의 가치](https://www.etri.re.kr/webzine/20190705/sub01.html)

DACON에 이와 관련된 대회가 있어 해당 대회에 참가하여 제공되는 데이터를 바탕으로 Deep Learning 기반의 성능예측 모델을 만들어 보겠습니다.
> [DACON - 자율주행 센서의 안테나 성능 예측 AI 경진대회](https://dacon.io/competitions/official/235927/overview/description)

## 논문탐색
* [제조 공정에서의 실시간 불량 탐지를 위한 딥러닝 모델 적용 연구](http://ktappi.kr/xml/30929/30929.pdf)
* [반도체 설비 센서 데이터를 활용한 딥러닝 기반의불량예측 모델에 관한 연구](https://www.koreascience.or.kr/article/CFKO202125036042269.pdf)
* [제조 공정에서 센서와 머신러닝을 활용한 불량예측 방안에 대한 연구](http://entrue.com/files/[4_2]%2089-98P_%EC%A0%9C%EC%A1%B0%20%EA%B3%B5%EC%A0%95_%ED%95%9C%EB%AC%B4%EB%AA%85%EC%B4%88.pdf)
