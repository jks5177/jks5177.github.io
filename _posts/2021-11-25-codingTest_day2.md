# 해시

프로그래머스 > 코딩테스트 연습 > 해시 > 전화번호 목록


```python
def solution(phone_book):
    answer = True
    d = {}
    
    for i in phone_book : # 길이에 따른 dictionary형 (key:길이, value:문자열)
        length = len(i)
        if length not in d.keys() :
            d[length] = [i]
        else :
            d[length].append(i)

    keys = sorted(list(d.keys()))[:-1]
    
    for i in phone_book : # 겹치는 값이 있는지 확인
        for j in keys :
            if len(i) > j :
                if i[:j] in d[j] :
                    answer = False
                    break
    return answer
```

정확도는 다 통과했으나 효율성에서 테스트4번을 실패했다.

아무리 생각해도 더 좋은 방법이 생각이 않나서 다른 사람의 풀이를 보았다.

## 다른사람 풀이


```python
def solution(phoneBook):
    phoneBook = sorted(phoneBook)

    for p1, p2 in zip(phoneBook, phoneBook[1:]):
        if p2.startswith(p1):
            return False
    return True
```

정렬을 하고 이전꺼와 다음꺼를 비교하는 간단한 방법이다...

나는 왜 해시 문제라고 해서 dictionary형을 사용하고 복잡하게 했을까하는 생각이 든다...

무조건 좋은 기술 최신 기술을 쓴다고 답이 아닌것 같다. 문제를 직관적으로 해석하고 푸는 방법이 가장 좋은 방법이라는 것을 다시 한번 배우고 간다.
