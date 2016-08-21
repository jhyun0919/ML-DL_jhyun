#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
* 어떻게 계산을 그래프화하는가.
* 어떻게 세션(Session)에서 그래프를 실행하는가.
* 어떻게 데이터를 텐서로 표현하는가.
* 어떻게 변수(Variable)로 상태를 관리하는가.
* 어떻게 임의의 연산(operations)으로부터 데이터를 얻거나 저장하기 위해 피드(feed)와 페치(fetch)를 사용하는가.
"""

# ==========================================================
#
# Making a Graph
#

import tensorflow as tf

# 1x2 형렬을 만드는 Constant 연산을 생성합니다.
# 이 연산자는 기본 그래프의 노드로 추가됩니다.
#
# 생성자에 의해 반환된 값(matrix1)은 Constant 연산의 출력을 나타냅니다.
matrix1 = tf.constant([[3., 3.]])

# 2x1 행렬을 만드는 또 다른 Constant 를 생성합니다.
matrix2 = tf.constant([[2.], [2.]])

# 'matrix1'과 'matrix2'를 입력으로 받는 Matmul 연산을 생성합니다.
# 반환값인 'product'는 행렬을 곱한 결과를 나타냅니다.
product = tf.matmul(matrix1, matrix2)

# ==========================================================
#
# Load Graph on Session
#

# 기본 그래프로 세션을 생성.
sess = tf.Session()

# matmul 연산을 실행하려면 matmul 연산의 출력을 나타내는 'product'를
# 인자로 넘겨주면서 세션의 'run()' 메소드를 호출합니다. 이렇게 호출하면
# matmul 연산의 출력을 다시 얻고 싶다는 뜻입니다.
#
# 연산이 필요로 하는 모든 입력은 세션에 의해 자동으로 실행됩니다.
# 일반적으로 병렬적으로 실행되지요.
#
# 따라서 'run(product)' 호출은 그래프 내의 세 개 연산을 실행하게 됩니다.
# 두 개의 상수와 matmul이 바로 그 연산이지요.
#
# 연산의 출력은 'result'에 numpy의 `ndarray` 객체 형태로 저장됩니다. 
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# 다 끝났으면 세션을 닫아줍니다.
sess.close()

#  "with" 블록을 이용해서 Session에 들어갈 수도 있습니다. 
#  이 경우 with 블록이 끝나면 Session이 자동으로 닫히게 됩니다.

with tf.Session() as sess:
    result = sess.run([product])
    print(result)

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)
        ...

# ==========================================================
#
# Interactive Usage
#

"""
이 문서에서 본 파이썬 예제에서는 그래프를 Session에 올린 뒤 Session.run() 메소드를 이용해 연산을 실행했습니다.

IPython과 같은 대화형 파이썬 환경에서 보다 쉽게 사용하려면,
InteractiveSession 클래스를 통해 Tensor.eval()과 Operation.run() 메소드를 호출할 수도 있습니다. 
그러면 세션을 위해 변수를 미리 생성할 필요가 없습니다.
"""

# 대화형 TensorFlow 세션에 들어갑니다.
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 'x'의 초기화 연산의 run() 메소드를 사용해서 'x'를 초기화합니다.
x.initializer.run()

# 'x'에서 'a'를 빼기 위해 연산을 추가. 실행하고 결과를 출력합니다.
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2. -1.]

# 다 끝났으면 세션을 닫아줍니다.
sess.close()

# ==========================================================
#
# Tensor
#

"""
TensorFlow 프로그램은 텐서(tensor) 데이터 구조를 사용해서 모든 데이터를 나타냅니다. 
계산 그래프에서 연산을 할 때에도 오직 텐서만 주고 받습니다. 
TensorFlow의 텐서는 n 차원의 배열이나 리스트라고 생각하면 됩니다. 
텐서는 정적 타입(static type), 랭크(rank), 모양(shape)의 속성을 가집니다.
"""

# ==========================================================
#
# Variables
#

import tensorflow as tf

# 변수를 하나 생성하고 스칼라 값인 0으로 초기화합니다.
state = tf.Variable(0, name="counter")

# one을 state에 더하는 연산을 생성합니다.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 변수는 그래프가 올라간 뒤 'init' 연산을 실행해서 반드시 초기화 되어야 합니다.
# 그 전에 먼저 'init' 연산을 그래프에 추가해야 합니다.
init_op = tf.initialize_all_variables()

# 그래프를 올리고 연산을 실행합니다.
with tf.Session() as sess:
    # 'init' 연산을 실행합니다.
    sess.run(init_op)
    # 'state'의 초기값을 출력합니다.
    print(sess.run(state))
    # 'state'를 갱신하는 연산을 실행하고 'state'를 출력합니다.
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# 출력:

# 0
# 1
# 2
# 3

# ==========================================================
#
# Fetches
#

"""
연산의 출력을 가져오기 위해서는 Session 객체에서 가져오고 싶은 텐서를 run()에 인자로 넘겨서 그래프를 실행해야 합니다. 
이전 예제에서 우리는 state라는 한 개의 노드를 가져왔지요. 
하지만 여러 개의 텐서를 가져올 수도 있습니다:
"""
import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)

# 출력:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

# ==========================================================
#
# Feeds
#

"""
지금까지 본 예제에서는 텐서의 값을 Conatants와 Variables에 미리 저장한 뒤 계산 그래프에 넘겨줍니다. 
TensorFlow는 그래프 내에서 직접 텐서의 값을 할당할 수 있는 방법도 제공하고 있습니다.

피드(feed)는 연산의 출력을 지정한 텐서 값으로 임시 대체합니다. 
임시로 사용할 피드 데이터는 run()의 인자로 넘겨줄 수 있습니다. 
피드 데이터는 run을 호출할 때 명시적 인자로 넘겨질 때만 사용됩니다. 
가장 흔한 사용법은 tf.placeholder()를 사용해 특정 연산을 "피드" 연산으로 지정하는 것입니다.
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))

# output:
# [array([ 14.], dtype=float32)]


import tensorflow as tf

# 변수를 0으로 초기화
state = tf.Variable(0, name="counter")

# state에 1을 더할 오퍼레이션 생성
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 그래프는 처음에 변수를 초기화해야 합니다. 아래 함수를 통해 init 오퍼레이션을 만듭니다.
init_op = tf.initialize_all_variables()

# 그래프를 띄우고 오퍼레이션들을 실행
with tf.Session() as sess:
    # 초기화 오퍼레이션 실행
    sess.run(init_op)
    # state의 초기 값을 출력
    print(sess.run(state))
    # state를 갱신하는 오퍼레이션을 실행하고, state를 출력
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
