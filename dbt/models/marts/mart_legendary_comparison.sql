-- Head-to-head stat comparison between legendary and non-legendary Pokémon.
with pokemon as (

    select * from {{ ref('int_pokemon_stats') }}

),

comparison as (

    select
        is_legendary,
        count(*)                            as pokemon_count,
        round(avg(base_stat_total), 1)      as avg_base_stat_total,
        round(avg(hp), 1)                   as avg_hp,
        round(avg(attack), 1)               as avg_attack,
        round(avg(defense), 1)              as avg_defense,
        round(avg(special_attack), 1)       as avg_special_attack,
        round(avg(special_defense), 1)      as avg_special_defense,
        round(avg(speed), 1)                as avg_speed,
        round(avg(total_offense), 1)        as avg_total_offense,
        round(avg(total_defense), 1)        as avg_total_defense

    from pokemon
    group by is_legendary

)

select * from comparison
