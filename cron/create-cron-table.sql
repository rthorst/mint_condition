/* SQL to create log results of nightly cron job */
create table if not exists cron (
	card_id serial primary key,
    url varchar,
    grade integer,
    title varchar,
    price numeric,
    confidence numeric,
    access_date_utc numeric
);

