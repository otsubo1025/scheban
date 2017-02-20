# -*- coding: utf-8 -*-
import datetime
from collections import defaultdict

import dateutil.parser
from application.google_calendar import get_upcoming_events


def extract_intervals(events_list):
    intervals = defaultdict(list)
    for events in events_list:
        for event in events:
            # Extract start and end time
            start_datetime_str = event['start'].get('dateTime', event['start'].get('date'))
            end_datetime_str = event['end'].get('dateTime', event['end'].get('date'))

            # Convert datetime str to datetime
            s_datetime = dateutil.parser.parse(start_datetime_str)
            e_datetime = dateutil.parser.parse(end_datetime_str)
            interval = (s_datetime, e_datetime)

            date = datetime.date(s_datetime.year, s_datetime.month, s_datetime.day)
            intervals[date].append(interval)

    return intervals


def add_sentinels(intervals, min_time, max_time):
    for date in intervals.keys():
        tzinfo = intervals[date][0][0].tzinfo
        s_datetime = datetime.datetime.combine(date, datetime.time.min.replace(tzinfo=tzinfo))
        e_datetime = datetime.datetime.combine(date, min_time.replace(tzinfo=tzinfo))
        intervals[date].append((s_datetime, e_datetime))
        s_datetime = datetime.datetime.combine(date, max_time.replace(tzinfo=tzinfo))
        e_datetime = datetime.datetime.combine(date, datetime.time.max.replace(tzinfo=tzinfo))
        intervals[date].append((s_datetime, e_datetime))
    return intervals


def sort_intervals(intervals):
    for date in intervals.keys():
        intervals[date].sort()
    return intervals


def find_free_intervals(intervals):
    free_intervals = defaultdict(list)
    for date, ivals in intervals.items():
        max_time = ivals[0][1]
        for i in range(len(ivals) - 1):
            first, second = ivals[i], ivals[i+1]
            max_time = max(max_time, first[1])
            if max_time >= second[0]:
                continue
            free_interval = (max_time, second[0])
            free_intervals[date].append(free_interval)
    return free_intervals


def get_available_time(min_time, max_time, *events_list):
    intervals = extract_intervals(events_list)
    intervals = add_sentinels(intervals, min_time, max_time)
    intervals = sort_intervals(intervals)
    intervals = find_free_intervals(intervals)
    return intervals


if __name__ == '__main__':
    calendar_ids = {'岡本': 's7c1eb9q7nj8l435ad407dnfds@group.calendar.google.com',
                    '山田': 'n3p38a1nrui275hkv4lm220dd0@group.calendar.google.com',
                    '貝原': '2uclqb6gtkk873iceeajn6n4q0@group.calendar.google.com'}
    john_events = get_upcoming_events(calendar_id=calendar_ids['岡本'], max_results=100)
    mary_events = get_upcoming_events(calendar_id=calendar_ids['山田'], max_results=100)
    mike_events = get_upcoming_events(calendar_id=calendar_ids['貝原'], max_results=100)

    from pprint import pprint
    min_time, max_time = datetime.time(9, 0), datetime.time(18, 0)  # search free time between 9 and 18 o'clock.
    pprint(get_available_time(min_time, max_time, john_events, mary_events, mike_events))