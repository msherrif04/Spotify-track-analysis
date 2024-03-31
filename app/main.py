import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance


def get_data():
    data = pd.read_csv("data/clean_data.csv", index_col=0)
    return data


def __calc_euclidean_distance(target_track, data_norm):
    data_result = pd.DataFrame()
    data_result["euclidean"] = [
        distance.euclidean(obj, target_track) for index, obj in data_norm.iterrows()
    ]
    data_result["track_id"] = data_norm.index

    return data_result


def add_sidebar():
    data = get_data()
    track_names = data["track_name"].sort_values(ascending=True).tolist()

    st.sidebar.header("Spotify Top Songs October 2022")

    selected = st.sidebar.selectbox("Select a song to get recommendations", track_names)

    return selected


def recommend_songs(data, selected_track):

    scaler = MinMaxScaler()

    numerical_cols = data.select_dtypes(include=np.number).columns

    data_norm = pd.DataFrame(
        scaler.fit_transform(data[numerical_cols]),
        columns=numerical_cols,
        index=data["track_id"],
    )

    # selected_track = "Knockin' On Heaven's Door"
    track_id = data[(data["track_name"] == selected_track)][["track_id"]]
    track_id = track_id.values[0][0]
    target_track = list(data_norm.loc[track_id])

    data_result = __calc_euclidean_distance(target_track, data_norm)

    data_rec = data_result.sort_values(by=["euclidean"]).iloc[:6]
    data_init = data.set_index(data.loc[:, "track_id"])
    track_list = pd.DataFrame()

    for i in list(data_rec.loc[:, "track_id"]):
        if i in list(data.loc[:, "track_id"]):
            track_info = data_init.loc[[i], ["track_name", "artists"]]
            track_list = pd.concat([track_list, track_info], ignore_index=True)

    recommendations = track_list.values.tolist()

    return recommendations, data_norm


def get_radar_chart(track_id, data_norm):
    input_data = data_norm.loc[track_id]
    categories = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    # Create the radar chart
    fig = go.Figure()

    # Add the traces
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["popularity"],
                input_data["duration_ms"],
                input_data["danceability"],
                input_data["energy"],
                input_data["loudness"],
                input_data["mode"],
                input_data["speechiness"],
                input_data["acousticness"],
                input_data["liveness"],
                input_data["valence"],
                input_data["tempo"],
            ],
            theta=categories,
            fill="toself",
            name="Vibe Map",
        )
    )

    # Update the layout
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        autosize=True,
    )

    return fig


def get_selected_track_id(selected_track, data):
    selected_track_id = data[(data["track_name"] == selected_track)][["track_id"]]
    selected_track_id = selected_track_id.iloc[0, 0]
    return selected_track_id


def main():
    st.set_page_config(
        page_title="Spotify song recommender",
        page_icon=":music:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with open("assets/styles.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # get the data
    data = get_data()

    # add sidebar
    selected_track = add_sidebar()

    with st.container():
        st.header("Spotify Music Vibe Map and Recommendation Engine")
        st.subheader(
            "Vibe into Your Perfect Playlist: Discover Music Tailored to Your Mood"
        )
        st.write(
            "Unlock your ideal soundtrack with our intuitive app, where music meets mood. Whether you're chilling, energized, or introspective, our algorithmic magic tailors playlists to match your vibe, ensuring every beat resonates with your moment."
        )

    # recommended songs
    recommendations, normalised_data = recommend_songs(data, selected_track)

    with st.container():
        col1, col2 = st.columns([3, 2])
        with col1:
            selected_track_id = get_selected_track_id(selected_track, data)
            radar_chart = get_radar_chart(selected_track_id, normalised_data)
            st.plotly_chart(radar_chart)

        with col2:
            st.subheader("Unlock Your Musical Journey - Your Melodic Matchmake")
            st.write(
                "Our app delivers personalized playlists tailored to your tastes, offering a seamless journey through a world of music. Let us be your musical companion, wherever you go. Don't miss out on the curated recommendations below — dive in and explore a universe of tunes waiting just for you."
            )
            st.write("You've just listened to: ")
            st.write(
                f"\n \t - <span class= selected_track > {recommendations[0][0]} - {recommendations[0][1]} </span>",
                unsafe_allow_html=True,
            )

    with st.container():

        st.title("You may enjoy these similar songs:")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            track_id = get_selected_track_id(recommendations[1][0], data)
            radar_chart = get_radar_chart(track_id, normalised_data)
            st.plotly_chart(radar_chart, use_container_width=True)
            st.write(
                f"\n \t'{recommendations[1][0]}-{recommendations[1][1]}'",
                unsafe_allow_html=True,
            )

        with col2:
            track_id = get_selected_track_id(recommendations[2][0], data)
            radar_chart = get_radar_chart(track_id, normalised_data)
            st.plotly_chart(radar_chart, use_container_width=True)
            st.write(f"\n \t -'{recommendations[2][0]}-{recommendations[2][1]}'")

        with col3:
            track_id = get_selected_track_id(recommendations[3][0], data)
            radar_chart = get_radar_chart(track_id, normalised_data)
            st.plotly_chart(radar_chart, use_container_width=True)
            st.write(f"\n \t -'{recommendations[3][0]}-{recommendations[3][1]}'")

        with col4:
            track_id = get_selected_track_id(recommendations[4][0], data)
            radar_chart = get_radar_chart(track_id, normalised_data)
            st.plotly_chart(radar_chart, use_container_width=True)
            st.write(f"\n \t -'{recommendations[4][0]}-{recommendations[4][1]}'")


if __name__ == "__main__":
    main()
