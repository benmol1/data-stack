with stg as (

    select * from {{ ref('stg_pokemon') }}

),

with_derived as (

    select
        *,

        -- Composite stat totals
        hp + attack + defense + special_attack + special_defense + speed
            as base_stat_total,

        attack + special_attack
            as total_offense,

        defense + special_defense
            as total_defense,

        -- Whether the Pokémon has a secondary type
        secondary_type is not null
            as is_dual_type

    from stg

)

select * from with_derived
