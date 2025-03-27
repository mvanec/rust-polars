use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use polars::{frame::DataFrame, io::SerReader, prelude::*};
use serde::Deserialize;
use serde_json::Value;
use tokio::{net::TcpListener, time::Instant};
use tracing::{error, info};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let now = Instant::now();
    info!("reading delays csv dataframe");

    let df_delays = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("../DBTrainrides.csv".into()))
        .expect("can read delays csv")
        .finish()
        .expect("can create delays dataframe");

    let df_cities = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("../plz_einwohner.csv".into()))
        .expect("can read cities csv")
        .finish()
        .expect("can create cities dataframe");

    info!("done reading csv. elapsed {:?}", now.elapsed());

    let app_state = Arc::new(AppState::new(df_delays, df_cities));

    let app = Router::new()
        .route("/delays", get(delays))
        .route("/cities/{zip}/population", get(cities_pop))
        .route("/cities/{zip}/area", get(cities_area))
        .route("/delays/{zip}/", get(delays_by_zip))
        .route("/delays/corr/{field1}/{field2}/", get(delays_corr))
        .with_state(app_state);

    let listener = TcpListener::bind("0.0.0.0:8000")
        .await
        .expect("can start web server on port 8000");

    info!("listening on {:?}", listener.local_addr());

    axum::serve(listener, app)
        .await
        .expect("can start web server");
}

pub struct AppState {
    pub df_delays: DataFrame,
    pub df_cities: DataFrame,
}

impl AppState {
    pub fn new(df_delays: DataFrame, df_cities: DataFrame) -> Self {
        Self {
            df_delays,
            df_cities,
        }
    }
}

#[derive(Deserialize, Eq, PartialEq)]
enum SortingOptions {
    Desc,
    Asc,
}

#[derive(Deserialize)]
struct DelayOptions {
    sorting: Option<SortingOptions>,
    limit: Option<u32>,
}

async fn delays(
    query: Query<DelayOptions>,
    app_state: State<Arc<AppState>>,
) -> Result<Json<Value>, StatusCode> {
    let mut sorting_mode = SortMultipleOptions::default()
        .with_order_descending(true)
        .with_nulls_last(true);
    if let Some(query_sorting) = &query.sorting {
        if *query_sorting == SortingOptions::Asc {
            sorting_mode = SortMultipleOptions::default()
                .with_order_descending(false)
                .with_nulls_last(true);
        }
    }
    let mut limit = 10;
    if let Some(query_limit) = query.limit {
        limit = query_limit
    }
    let delays = app_state
        .df_delays
        .clone()
        .lazy()
        .sort(["arrival_delay_m"], sorting_mode)
        .select([cols([
            "ID",
            "arrival_delay_m",
            "departure_delay_m",
            "city",
            "zip",
        ])])
        .limit(limit)
        .collect()
        .map_err(|e| {
            error!("could not query delay times: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    df_to_json(delays)
}

async fn delays_by_zip(
    Path(zip): Path<String>,
    app_state: State<Arc<AppState>>,
) -> Result<Json<Value>, StatusCode> {
    let delays_for_zip = app_state
        .as_ref()
        .df_delays
        .clone()
        .lazy()
        .join(
            app_state.as_ref().df_cities.clone().lazy(),
            [col("zip")],
            [col("zip")],
            JoinArgs::new(JoinType::Inner),
        )
        .group_by(["zip"])
        .agg([
            len().alias("len"),
            col("arrival_delay_m")
                .cast(DataType::Int64)
                .sum()
                .alias("sum_arrival_delays"),
            col("arrival_delay_m").alias("max").max(),
            col("pop").unique().first(),
            col("city").unique().first(),
        ])
        .filter(col("zip").str().contains_literal(lit(zip.as_str())))
        .collect()
        .map_err(|e| {
            error!("could not query delay times by zip {}: {}", zip, e);
            StatusCode::BAD_REQUEST
        })?;

    df_to_json(delays_for_zip)
}

async fn delays_corr(
    Path((field_1, field_2)): Path<(String, String)>,
    app_state: State<Arc<AppState>>,
) -> Result<Json<Value>, StatusCode> {
    let corr = app_state
        .as_ref()
        .df_delays
        .clone()
        .lazy()
        .join(
            app_state.as_ref().df_cities.clone().lazy(),
            [col("zip")],
            [col("zip")],
            JoinArgs::new(JoinType::Inner),
        )
        .select([
            len(),
            pearson_corr(col(&field_1), col(&field_2))
                .alias(&format!("corr_{}_{}", field_1, field_2)),
        ])
        .collect()
        .map_err(|e| {
            error!(
                "could not query delay time correlation between {} and {}: {}",
                field_1, field_2, e
            );
            StatusCode::BAD_REQUEST
        })?;

    df_to_json(corr)
}

fn df_to_json(df: DataFrame) -> Result<Json<Value>, StatusCode> {
    serde_json::to_value(&df)
        .map_err(|e| {
            error!("could not serialize dataframe to json: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
        .map(|v| v.into())
}

// pub fn df_to_json(df: DataFrame) -> Result<Json<Value>, StatusCode> {
//     let mut buffer = Vec::new();

//     JsonWriter::new(&mut buffer)
//         .with_json_format(JsonFormat::Json)
//         .finish(&mut df.clone())
//         .map_err(|e| {
//             error!("Failed to write DataFrame to JSON: {}", e);
//             StatusCode::INTERNAL_SERVER_ERROR
//         })?;

//     let json_string = String::from_utf8(buffer).map_err(|e| {
//         error!("Failed to convert buffer to UTF-8: {}", e);
//         StatusCode::INTERNAL_SERVER_ERROR
//     })?;

//     serde_json::from_str(&json_string)
//         .map_err(|e| {
//             error!("Failed to parse JSON string: {}", e);
//             StatusCode::INTERNAL_SERVER_ERROR
//         })
//         .map(Json)
// }

fn cities_zip_to(df_cities: &DataFrame, zip: &str, field: &str) -> Result<DataFrame, StatusCode> {
    df_cities
        .clone()
        .lazy()
        .filter(col("zip").str().contains_literal(lit(zip)))
        .select([col("zip"), col(field)])
        .collect()
        .map_err(|e| {
            error!("could not query {} for zip: {}", field, e);
            StatusCode::BAD_REQUEST
        })
}

async fn cities_pop(
    Path(zip): Path<String>,
    app_state: State<Arc<AppState>>,
) -> Result<Json<Value>, StatusCode> {
    let pop_for_zip = cities_zip_to(&app_state.as_ref().df_cities, &zip, "pop")?;
    df_to_json(pop_for_zip)
}

async fn cities_area(
    Path(zip): Path<String>,
    app_state: State<Arc<AppState>>,
) -> Result<Json<Value>, StatusCode> {
    let area_for_zip = cities_zip_to(&app_state.as_ref().df_cities, &zip, "area")?;
    df_to_json(area_for_zip)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df_to_json_t(df: DataFrame) -> Result<Json<Value>, StatusCode> {
        serde_json::to_value(&df)
            .map_err(|e| {
                error!("could not serialize dataframe to json: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })
            .map(|v| v.into())
    }

    #[test]
    fn test_df_to_json() -> Result<(), StatusCode> {
        let df = df! {
            "a" => [1,2,3,4,5],
            "b" => ["a","b","c","d","e"],
        }
        .unwrap();
        let df_json = df_to_json_t(df);

        match df_json {
            Ok(json) => println!("{}", serde_json::to_string_pretty(&json.0).unwrap()),
            Err(status) => println!("Error: {}", status)
        }
        Ok(())
    }

    #[test]
    fn test_df_serialization() -> Result<(), StatusCode> {
        let df = df! {
            "a" => [1,2,3,4,5],
            "b" => ["a","b","c","d","e"],
        }
        .unwrap();
        let df_json = df_to_json(df)?;
        println!("df_json {:?}", df_json);
        Ok(())
    }
}
