# -*- coding: utf-8 -*-
from slackbot.bot import respond_to, listen_to
import datetime # datetimeモジュールのインポート
import locale   # import文はどこに書いてもOK(可読性などの為、慣例でコードの始めの方)
import re


# 予定の入力
@listen_to('(.*)をメモ')
def input(message, something, ver=None):
    # today()メソッドで現在日付・時刻のdatetime型データの変数を取得
    d = datetime.datetime.today()
    # ファイルに書き込み
    f = open('test.txt', 'a', encoding='utf-8')
    f.writelines(something + '(入力日：' + str(d.month) + '/' + str(d.day) + ')' + '\n')
    f.writelines("")
    message.reply('入力を完了しました')
    message.reply('{0}'.format(something))


# 予定の出力
@listen_to('(.*)の予定を教えて')
def output(message, something, ver=None):
    d = datetime.datetime.today()
    # ファイルの読み込み
    text = ''
    f = open('test.txt','r', encoding='utf-8')
    lines2 = f.readlines()
    f.close()
    if something != '全て':
        for line in lines2:
            # print(line)
            # 入力されたメッセージのカテゴリと一致するものを探索
            # 入力されたメッセージの日付と一致するものを探索
            s = line.find(something)
            if s != -1:
                message.reply(line)
    else:
        for line in lines2:
            print(line)
            message.reply(line)
    print
    # 出力


# 予定の削除
@listen_to('(.*)の予定を削除')
def deletememo(message, something, ver=None):
    d = datetime.datetime.today()
    # f = open('test.txt', 'r+', encoding='utf-8')
    # lines = f.readlines()
    # f.close()
    lis = []
    for line in open("test.txt", 'r+', encoding='utf-8').readlines():
        print(line)
        s = line.find(something)
        if s is -1:
            lis.append(line)
            print(s)
    f = open('test.txt', 'w', encoding='utf-8')
    for line in lis:
        print(line)
        f.writelines(line)
        # f.writelines("")
    message.reply('{0}の予定を削除しました'.format(something))
    # 削除
