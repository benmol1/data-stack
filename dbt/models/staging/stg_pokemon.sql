with source as (

    select * from {{ source('raw', 'pokemon_data') }}

),

renamed as (

    select
        -- Identifiers
        cast("Num" as integer)              as pokemon_id,
        "Name"                              as pokemon_name,

        -- Typing
        "Type1"                             as primary_type,
        nullif("Type2", '')                 as secondary_type,

        -- Base stats
        cast("HP" as integer)               as hp,
        cast("Attack" as integer)           as attack,
        cast("Defense" as integer)          as defense,
        cast("SpAtk" as integer)            as special_attack,
        cast("SpDef" as integer)            as special_defense,
        cast("Speed" as integer)            as speed,

        -- Metadata
        cast("Generation" as integer)       as generation,
        "Legendary" = 'TRUE'                as is_legendary

    from source

)

select * from renamed
