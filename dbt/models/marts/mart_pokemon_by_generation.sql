-- Per-generation summary: how stat power and legendary density changed over time.
with pokemon as (

    select * from {{ ref('int_pokemon_stats') }}

),

by_generation as (

    select
        generation,
        count(*)                                                        as pokemon_count,
        sum(case when is_legendary then 1 else 0 end)                  as legendary_count,
        round(
            100.0 * sum(case when is_legendary then 1 else 0 end)
            / count(*), 1
        )                                                               as legendary_pct,
        round(avg(base_stat_total), 1)                                  as avg_base_stat_total,
        max(base_stat_total)                                            as max_base_stat_total,
        round(avg(total_offense), 1)                                    as avg_total_offense,
        round(avg(total_defense), 1)                                    as avg_total_defense,
        round(avg(speed), 1)                                            as avg_speed

    from pokemon
    group by generation
    order by generation

)

select * from by_generation
