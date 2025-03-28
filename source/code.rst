加密&解密
======================

对称加密
---------------------------------------------------------------------

使用同一秘钥加解密,秘钥只有通信双方有(DES,3DES,AES)

- 按位异或:相同为0,不同为1 

- ECB 电子密码本模式:对字节分组不够时,需要对明文进行填充,容易被破解

- CBC 密码块链:对字节分组不够时,需要对明文进行填充,需要初始化向量

- CFB 密文反馈:需要初始化向量,不需要对明文填充(初始化向量(iv)加密后,明文1组与加密后的密文按位异或,然后将加密后的明文1组与明文2组按位异或,依次往后,不需要对明文填充)

- OFB 输出反馈:需要初始化向量,不需要对明文进行填充(不断对初始化向量进行加密,用于加密明文分组异或)

- CTR 计数模式:不需要初始化向量(但是需要初始化种子),不需要对明文进行填充,效率最高(将明文分组对应的计数器加密后与明文异或)

| DES

1. DES的密钥长度是64比特,每隔7bit会设置一个用于错误检查的比特,因此实质上其密钥长度是56bit

- 分组长度:8字节

- 秘钥长度:8字节

| AES

- 分组长度:16字节

- 秘钥长度:16字节(go语言中存在的),24,32

对称加密流程

1. 创建一个底层使用des,3des,aes接口

2. 准备加密条件:初始化向量、对明文填充

3. 创建分组模式CTR

4. 加密,得到密文


哈希函数
--------------------------------

 - md5/md4 散列值：16字节
 - sha1 散列值:20字节
 - sha224 散列值:28字节
 - sha256 32
 - sha384 48
 - sha512 64


非对称加密
--------------------------------------------------------

使用不同秘钥对数据进行加解密

| 公钥:可以分发给任何人
| 私钥:只有自己拥有,数据对谁重要私钥就在谁手里

- RSA加密

- 单向散列(哈希算法):无法还原原始数据

- ECC椭圆曲线加密


消息认证码(Hmac)
-----------------------------

目的:保证数据的完整性及一致性

存在问题

- 秘钥分发困难,使用非对称加密解决秘钥分发

- 不能进行第三方证明

- 不能防止否认

数字签名
---------------------------------------------

解决了消息认证码的弊端,但是无法判断公钥的归属

1. 签名

 - 原始数据进行哈希运算 —> 散列值

 - 使用非对称私钥对散列值加密 —> 加密

 - 将数据和签名发送

2. 验证

 - 接收数据

 - 数字签名用公钥解密

 - 对原始数据进行哈希运算得到散列值,进行比较

签名的方式:RSA,椭圆曲线签名

 - 生成密钥对

 - 分发公钥

 - 签名的人
   - 读私钥
   - pem解码
   - x509解码 -> 私钥结构体
   - 对原始数据进行哈希
   - 私钥加密哈希散列值

 - 验证签名



https通信流程
-------------------------------------

准备工作：服务器端需要CA证书、生成非对称密钥对

1. 客户端连接服务器(通过域名)

	- 域名绑定IP地址
	- 服务器解析域名成IP地址
2. 服务器收到客户端请求

	- 服务器将CA证书发送到客户端的浏览器
3. 客户端拿到服务器的证书

	- 验证证书的签发机构
	- 服务器的公钥
	- 验证域名
	- 验证证书的有效期
4. 客户端产生随机数并用服务器的公钥加密(作为对称加密的密钥)
5. 服务器接收文件并使用私钥解密
6. 客户端通过对称加密对数据加密






