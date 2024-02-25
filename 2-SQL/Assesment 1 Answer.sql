
--Q1
select * from cd.facilities;

--Q2


--Q3
select name,membercost from cd.facilities
where membercost>0;

--Q4
select facid,name,membercost,monthlymaintenance from cd.facilities
where membercost>0 and membercost<(monthlymaintenance/50);

--Q5
select * from cd.facilities 
where name like '%Tennis%';

--Q6
select * from cd.facilities
where facid in (1,5);

--Q7
select memid,surname,firstname,joindate from cd.members
where joindate > '2012-09-01';

--Q8
select * from cd.members;

select distinct(surname) from cd.members
order by surname asc
limit 10; 

--Q9
select joindate from cd.members
order by joindate desc
limit 1;

--Q10
select * from cd.facilities;

select count(facid) from cd.facilities 
where guestcost>10;

--Q11
select * from cd.bookings;
select facid,sum(slots) as TotalSlots from cd.bookings
where starttime between '2012-09-01' and '2012-10-01'
group by facid;

--Q12
select facid, sum(slots) as TotalSlots from cd.bookings
group by facid 
having sum(slots) > 1000
order by facid asc;

--Q13 
select * from cd.bookings;

select starttime as start, name 
from cd.bookings inner join cd.facilities
on cd.bookings.facid = cd.facilities.facid
where name like 'Tennis Court%' and starttime between '2012-09-21' and '2012-09-22'
order by starttime;

--Siranch aahe he khalch
select cd.bookings.starttime, cd.facilities.name
from cd.facilities
inner join cd.bookings
on cd.bookings.facid = cd.facilities.facid
where cd.facilities.facid in (0,1)
and cd.bookings.starttime >='2012-09-21'
and cd.bookings.starttime < '2012-09-22'
order by cd.bookings.starttime;

--Q14
select firstname from cd.members;
select starttime from cd.bookings
inner join cd.facilities 
on cd.bookings.facid = cd.facilities.facid
where memid in (select memid from cd.members where firstname like 'David' and surname like 'Farrell');

