-- Aggregated stats per primary type, useful for comparing type strength profiles.
with pokemon as (

    select * from {{ ref('int_pokemon_stats') }}

),

by_type as (

    select
        primary_type,
        count(*)                                                as pokemon_count,
        sum(case when is_legendary then 1 else 0 end)          as legendary_count,
        round(avg(base_stat_total), 1)                         as avg_base_stat_total,
        max(base_stat_total)                                    as max_base_stat_total,
        round(avg(hp), 1)                                      as avg_hp,
        round(avg(attack), 1)                                   as avg_attack,
        round(avg(defense), 1)                                  as avg_defense,
        round(avg(special_attack), 1)                          as avg_special_attack,
        round(avg(special_defense), 1)                         as avg_special_defense,
        round(avg(speed), 1)                                    as avg_speed

    from pokemon
    group by primary_type
    order by avg_base_stat_total desc

)

select * from by_type
