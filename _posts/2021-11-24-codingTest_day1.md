# 해시

프로그래머스 > 코딩테스트 연습 > 해시 > 완주하지 못한 선수


```python
def solution(participant, completion):
    answer = ''
    d = {}
    for x in participant :
        if x not in d.keys() :
            d[x] = 1
        else :
            d[x] += 1
            
    for x in completion :
        d[x] -= 1
    
    answer = list(d.keys())[list(d.values()).index(1)]
    
    return answer
```

해시에 대한 문제로 Key : Value를 사용하는 Dictionary형을 사용하였다.

처음에는 for문을 사용하여 진행하였지만 테스트 케이스에서 시간 오류가 떠서 고민하던 중 질문하기에서 해시 문제라는 것을 깨닮고 Dictionary를 사용해 진행했습니다.

## 다른 사람의 풀이


```python
import collections

def solution(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]
```

다른 사람은 collections 라이브러리를 사용해서 더 간결하게 진행했다.

collections 라이브러리에 대해 찾아보고 코드를 해석하는 시간을 갖도록하겠습니다.

### collections

참고 자료 : https://docs.python.org/ko/3/library/collections.html

이 모듈은 파이썬의 범용 내장 컨테이너 dict, list, set 및 tuple에 대한 대안을 제공하는 특수 컨테이너 데이터형을 구현합니다.

* namedtuple() : 이름 붙은 필드를 갖는 튜풀 서브 클래스를 만들기 위한 팩토리 함수

* dque : 양쪽 끝에서 빠르게 추가와 삭제를 할 수 있는 리스트류 컨테이너

* ChainMap : 여러 매핑의 단일 뷰를 만드는 딕셔너리류 클래스

* Counter : 해시 가능한 객체를 세는 데 사용하는 딕셔너리 서브 클래스

* OrderdDict : 항목이 추가된 순서를 기억하는 딕셔너리 서브 클래스

* defauldict : 누락된 값을 제공하기 위해 팩토리 함수를 호출하는 딕셔너리 서브 클래스

* UserDict : 더 쉬운 딕셔너리 서브 클래싱을 위해 딕셔너리 객체를 감싸는 래퍼

* UserList : 더 쉬운 리스트 서브 클래싱을 위해 리스트 객체를 감싸는 래퍼

* UserString : 더 쉬운 문자열 서브 클래싱을 위해 문자열 객체를 감싸는 래퍼

### Counter

Conter는 해시 간으한 객체를 세기 위한 dict 서브 클래스입니다. 요소가 딕셔너리 키로 저장되고 개수가 딕셔너리값으로 저장되는 컬렉션입니다. 개수는 0이나 음수를 포함하는 임의의 정숫값이 될 수 있습니다. Counter 클래스는 다른 언어의 백(bag)이나 멀티 셋(multiset)과 유사합니다.

요소는 이터러블로부터 계산되거나 다른 매핑(또는 계수기)에서 초기화됩니다.


```python
from collections import Counter
c = Counter() # a new, empty counter
print('c = Counter()','\n',c)
c = Counter('gallahad') # a new counter from an iterable
print("c = Counter('gallahad')",'\n',c)
c = Counter({'red':4, ' blue':2}) # a new counter from a mapping
print("c = Counter({'red':4, ' blue':2})",'\n', c)
c = Counter(cats=4, dogs=8) # a new counter from keyword args
print("c = Counter(cats=4, dogs=8)",'\n',c)
```

    c = Counter() 
     Counter()
    c = Counter('gallahad') 
     Counter({'a': 3, 'l': 2, 'g': 1, 'h': 1, 'd': 1})
    c = Counter({'red':4, ' blue':2}) 
     Counter({'red': 4, ' blue': 2})
    c = Counter(cats=4, dogs=8) 
     Counter({'dogs': 8, 'cats': 4})
    

계수기 객체는 누락된 항목에 대해 KeyError를 발생시키는 대신 0을 반환한다는 점을 제외하고 딕셔너리 인터페이스를 갖습니다.


```python
c = Counter(['eggs', 'ham'])
c['bacon'] # count of a missing element is zero
```




    0



개수를 0으로 설정해도 계수기에서 요소가 제거되지 않습니다. 완전히 제거하려면 del을 사용하십시오.


```python
c['sausage']=0
del c['sausage']
```
