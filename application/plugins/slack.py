# -*- coding: utf-8 -*-
import sys
from slackbot.bot import respond_to
sys.path.append('..')
from google_calendar import events2text
#from find_avairable_time import get_available_time

@respond_to('今後の予定を教えて')
def respond_schedule(message):
    calendar_id1 = 's7c1eb9q7nj8l435ad407dnfds@group.calendar.google.com'
    #calendar_id2 = 'n3p38a1nrui275hkv4lm220dd0@group.calendar.google.com'
    #calendar_id3 = '2uclqb6gtkk873iceeajn6n4q0@group.calendar.google.com'
    reply_message = events2text(calendar_id=calendar_id1) #+ events2text(calendar_id=calendar_id2) + events2text(calendar_id=calendar_id3)
    #reply_massage =
    message.reply(reply_message)
