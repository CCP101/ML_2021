from datetime import datetime
from dateutil.parser import parse

now = datetime.now()
time_ZZ = parse('2021-12-25/08:30:00')
time_EG = parse('2021-12-25/14:00:00')
time_Z1 = parse('2021-12-26/08:30:00')
time_Z2 = parse('2021-12-26/14:00:00')

a = time_ZZ - now
a_day = a.days
a_hour = int(a.seconds / 3600)
a_minute = int((a.seconds % 3600) / 60)
a_second = int((a.seconds % 3600) % 60)
print("距离政治考试还有{}天{}小时{}分{}秒".format(a_day, a_hour, a_minute, a_second))
a = time_EG - now
a_day = a.days
a_hour = int(a.seconds / 3600)
a_minute = int((a.seconds % 3600) / 60)
a_second = int((a.seconds % 3600) % 60)
print("距离英语考试还有{}天{}小时{}分{}秒".format(a_day, a_hour, a_minute, a_second))
a = time_Z1 - now
a_day = a.days
a_hour = int(a.seconds / 3600)
a_minute = int((a.seconds % 3600) / 60)
a_second = int((a.seconds % 3600) % 60)
print("距离专业课1考试还有{}天{}小时{}分{}秒".format(a_day, a_hour, a_minute, a_second))
a = time_Z2 - now
a_day = a.days
a_hour = int(a.seconds / 3600)
a_minute = int((a.seconds % 3600) / 60)
a_second = int((a.seconds % 3600) % 60)
print("距离专业课2考试还有{}天{}小时{}分{}秒".format(a_day, a_hour, a_minute, a_second))