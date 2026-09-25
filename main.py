import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



# Load data
df = pd.read_csv("players_cleaned.csv")



# Position based features

general_features = [
    "Gls_per90",
    "Ast_per90",
    "xG_per90",
    "xAG_per90",
    "PrgP_per90",
    "PrgC_per90",
    "KP_per90",
    "Tkl_per90",
    "Int_per90",
    "Clr_per90",
    "Carries_per90",
    "Touches_per90"
]


forward_features = [
    "Gls_per90",
    "Ast_per90",
    "xG_per90",
    "xAG_per90",
    "Sh_per90",
    "SoT_per90",
    "KP_per90",
    "SCA_per90",
    "GCA_per90",
    "PrgC_per90",
    "PrgR_per90",
    "CPA_per90"
]


midfielder_features = [
    "Gls_per90",
    "Ast_per90",
    "xG_per90",
    "xAG_per90",
    "PrgP_per90",
    "KP_per90",
    "PPA_per90",
    "Tkl_per90",
    "Int_per90",
    "Carries_per90",
    "Touches_per90",
    "Rec_per90"
]


defender_features = [
    "Gls_per90",
    "Ast_per90",
    "PrgP_per90",
    "PrgC_per90",
    "Tkl_per90",
    "Int_per90",
    "Clr_per90",
    "Blocks_per90",
    "Touches_per90",
    "Carries_per90",
    "Rec_per90",
    "Won_per90"
]


# Full stats names

stat_names = {
    "Gls_per90": "Goals/90",
    "Ast_per90": "Assists/90",
    "xG_per90": "xG/90",
    "xAG_per90": "xAG/90",
    "PrgP_per90": "Progressive Passes/90",
    "PrgC_per90": "Progressive Carries/90",
    "KP_per90": "Key Passes/90",
    "Tkl_per90": "Tackles/90",
    "Int_per90": "Interceptions/90",
    "Clr_per90": "Clearances/90",
    "Touches_per90": "Touches/90",
    "Carries_per90": "Carries/90",
    "Sh_per90": "Shots/90",
    "SoT_per90": "Shots on Target/90",
    "PPA_per90": "Passes into Penalty Area/90",
    "SCA_per90": "Shot-Creating Actions/90",
    "GCA_per90": "Goal-Creating Actions/90",
    "PrgR_per90": "Progressive Receives/90",
    "CPA_per90": "Carries into Penalty Area/90",
    "Blocks_per90": "Blocks/90",
    "Rec_per90": "Successful Receives/90",
    "Won_per90": "Aerial Duels Won/90"
}


def find_player(df, player_name):

    player = df[
        df["Player"].str.lower() == player_name.lower()
    ]

    if player.empty:
        return None

    return player


def get_position_features(position):

    primary_position = position.split(",")[0]

    if primary_position == "DF":
        return defender_features

    elif primary_position == "MF":
        return midfielder_features

    elif primary_position == "FW":
        return forward_features

    else:
        return None
    

def choose_feature_set():

    print()
    print("Choose feature set:")
    print("1. General")
    print("2. Forward")
    print("3. Midfielder")
    print("4. Defender")
    print("5. Back to main menu")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        return "General", general_features

    elif choice == "2":
        return "Forward", forward_features

    elif choice == "3":
        return "Midfielder", midfielder_features

    elif choice == "4":
        return "Defender", defender_features

    else:
        return None, None


def get_player_names(df):

    player_names = []

    while True:

        player_name = input(
            f"Player {len(player_names) + 1}: "
        ).strip()

        if player_name == "":
            break

        player = find_player(df, player_name)

        if player is None:

            print("Player not found. Please try again.")

            continue

        if player_name.lower() in [
            name.lower() for name in player_names
        ]:

            print("You have already selected this player.")

            continue

        player_names.append(player_name)

    return player_names


# Compare players

def compare_players(df):

    print()
    print("-" * 40)
    print("          PLAYER COMPARISON")
    print("-" * 40)

    print()
    print("Enter player names one at a time.")
    print("Press ENTER on an empty line when finished.")
    print()

    player_names = get_player_names(df)

    if not player_names:

        print("\nNo players selected.")

        return

    # Choose which features to compare
    feature_name, features = choose_feature_set()

    if features is None:

        print("\nReturning to main menu.")

        return

    selected_players = []

    for player_name in player_names:

        player = find_player(df, player_name)

        selected_players.append(player)

    comparison_df = pd.concat(
        selected_players,
        ignore_index=True
    )

    columns_to_show = [
        "Player",
        "Pos",
        "Squad",
        "Min"
    ] + features

    comparison_df = comparison_df[columns_to_show]

    stat_df = comparison_df.drop(
        columns=[
            "Player",
            "Pos",
            "Squad",
            "Min"
        ]
    )

    comparison_table = stat_df.T

    comparison_table.columns = comparison_df["Player"]

    comparison_table.index = comparison_table.index.map(
        stat_names
    )

    comparison_table = comparison_table.round(2)

    print()
    print("-" * 50)
    print(f"{feature_name.upper()} PLAYER COMPARISON")
    print("-" * 50)

    print(comparison_table)

    print()
    print("Returning to main menu.")


def plot_similar_players(
    player,
    similar_players,
    features
):

    players_to_plot = pd.concat(
        [player, similar_players],
        ignore_index=True
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        players_to_plot[features]
    )

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=features
    )

    x = np.arange(len(features))

    width = 0.8 / len(players_to_plot)

    for i in range(len(players_to_plot)):

        offset = (
            i - (len(players_to_plot) - 1) / 2
        ) * width

        plt.bar(
            x + offset,
            scaled_df.iloc[i],
            width,
            label=(
                players_to_plot.iloc[i]["Player"]
                + " ("
                + players_to_plot.iloc[i]["Pos"]
                + ")"
            )
        )

    labels = [
        stat_names[feature]
        for feature in features
    ]

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right"
    )

    plt.ylabel("Standardised value")

    plt.title(
        "Statistical Profiles of Similar Players"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# Find similar players

def find_similar_players(df):

    print()
    print("-" * 40)
    print("        FIND SIMILAR PLAYERS")
    print("-" * 40)

    player_name = input(
        "\nEnter player name: "
    ).strip()

    player = find_player(df, player_name)

    if player is None:

        print("Player not found.")

        return

    position = player["Pos"].iloc[0]

    if "GK" in position:

        print(
            "\nGoalkeepers are not currently supported."
        )

        return

    print()
    print("Compare against:")
    print("1. All players")
    print("2. Same position")

    comparison_choice = input(
        "\nSelect option: "
    ).strip()

    if comparison_choice == "1":

        features = general_features

        candidates = df[
            ~df["Pos"].str.contains("GK")
        ].copy()

        print(
            "\nUsing general player features..."
        )

    elif comparison_choice == "2":

        features = get_position_features(
            position
        )

        if features is None:

            print(
                "Could not determine player position."
            )

            return

        primary_position = position.split(",")[0]

        candidates = df[
            df["Pos"].str.contains(
                primary_position
            )
        ].copy()

        print(
            f"\nUsing {primary_position} "
            "specific features..."
        )

    else:

        print("Invalid option.")

        return


    candidates = candidates[
        candidates["Player"].str.lower()
        != player_name.lower()
    ]

    # Standardise the candidate stats
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        candidates[features]
    )

    player_scaled = scaler.transform(
        player[features]
    )

    # Calculate Euclidean distance

    distances = []

    for i in range(len(candidates)):

        candidate = scaled_data[i]

        distance = np.linalg.norm(
            player_scaled[0] - candidate
        )

        distances.append(distance)

    candidates["distance"] = distances

    similar_players = candidates.sort_values(
        "distance"
    )

    top_players = similar_players.head(5)


    print()
    print("-" * 55)
    print(
        f"       SIMILAR PLAYERS TO "
        f"{player_name.upper()}"
    )
    print("-" * 55)

    for i, (_, row) in enumerate(
        top_players.iterrows(),
        start=1
    ):

        print(
            f"{i}. {row['Player']:<25} "
            f"({row['Pos']}) "
            f"Distance: {row['distance']:.3f}"
        )


    more_detail = input(
        "\nWould you like more detail? (y/n): "
    ).strip().lower()

    if more_detail == "y":

        detail_players = pd.concat(
            [
                player,
                top_players
            ],
            ignore_index=True
        )

        detail_data = detail_players[
            ["Player", "Pos"] + features
        ].copy()

        detail_data["Player"] = (
            detail_data["Player"]
            + " ("
            + detail_data["Pos"]
            + ")"
        )

        detail_data = detail_data.drop(
            columns=["Pos"]
        )

        detail_table = detail_data.set_index(
            "Player"
        ).T

        detail_table.index = [
            stat_names[feature]
            for feature in detail_table.index
        ]

        detail_table = detail_table.round(2)

        print()
        print("-" * 50)
        print("             PLAYER DETAILS")
        print("-" * 50)

        print(detail_table)

        show_chart = input(
            "\nWould you like to see this data "
            "as a bar chart? (y/n): "
        ).strip().lower()

        if show_chart == "y":

            plot_similar_players(
                player,
                top_players,
                features
            )

    print()
    print("Returning to main menu.")


# Clustering

def choose_cluster_type():

    print()
    print("-" * 40)
    print("       PLAYER CLUSTER ANALYSIS")
    print("-" * 40)

    print()
    print("1. General")
    print("2. Forward")
    print("3. Midfielder")
    print("4. Defender")
    print("5. Back to main menu")

    choice = input(
        "\nSelect option: "
    ).strip()

    if choice == "1":

        players = df[
            ~df["Pos"].str.contains("GK")
        ].copy()

        return "General", players, general_features

    elif choice == "2":

        players = df[
            df["Pos"].str.contains("FW")
        ].copy()

        return "Forward", players, forward_features

    elif choice == "3":

        players = df[
            df["Pos"].str.contains("MF")
        ].copy()

        return "Midfielder", players, midfielder_features

    elif choice == "4":

        players = df[
            df["Pos"].str.contains("DF")
        ].copy()

        return "Defender", players, defender_features

    else:

        return None, None, None


def show_cluster_statistics(
    clustered_players,
    features,
    number_of_clusters
):

    cluster_stats = clustered_players.groupby(
        "Cluster"
    )[features].mean()

    cluster_sizes = clustered_players[
        "Cluster"
    ].value_counts().sort_index()

    print()
    print("-" * 60)
    print("                 CLUSTER STATISTICS")
    print("-" * 60)

    for cluster_number in range(
        number_of_clusters
    ):

        print()
        print(
            f"Cluster {cluster_number}"
        )

        print(
            f"Players: "
            f"{cluster_sizes[cluster_number]}"
        )

        stats = cluster_stats.loc[
            cluster_number
        ]

        stats_table = pd.DataFrame({
            "Average Stats": stats
        })

        stats_table.index = [
            stat_names[feature]
            for feature in stats_table.index
        ]

        stats_table = stats_table.round(2)

        print(stats_table)


def show_cluster_players(
    clustered_players,
    number_of_clusters
):

    print()
    print("-" * 60)
    print("               PLAYERS IN CLUSTERS")
    print("-" * 60)

    for cluster_number in range(
        number_of_clusters
    ):

        cluster_players = clustered_players[
            clustered_players["Cluster"]
            == cluster_number
        ]

        cluster_players = cluster_players.sort_values(
            "Player"
        )

        print()
        print(
            f"Cluster {cluster_number}"
        )

        for _, row in cluster_players.iterrows():

            print(
                f"- {row['Player']} "
                f"({row['Pos']})"
            )


def find_player_cluster(
    clustered_players
):

    while True:

        player_name = input(
            "\nEnter player name "
            "(or press ENTER to go back): "
        ).strip()

        if player_name == "":
            return

        player = find_player(
            clustered_players,
            player_name
        )

        if player is None:

            print(
                "Player is not in this cluster analysis."
            )

            print(
                "Please try another player."
            )

            continue

        row = player.iloc[0]

        cluster_number = row["Cluster"]

        print()
        print(
            f"{row['Player']} "
            f"({row['Pos']}) belongs to:"
        )

        print(
            f"Cluster {cluster_number}"
        )

        # Show other players in the cluster
        other_players = clustered_players[
            (clustered_players["Cluster"] == cluster_number)
            &
            (
                clustered_players["Player"].str.lower()
                != row["Player"].lower()
            )
        ]

        print()
        print("Other players in this cluster:")

        for _, other in other_players.iterrows():

            print(
                f"- {other['Player']} "
                f"({other['Pos']})"
            )

        input(
            "\nPress ENTER to return..."
        )

        return


# PCA visualisation

def show_cluster_pca(
    clustered_players,
    features
):

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        clustered_players[features]
    )

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(
        scaled_data
    )

    plt.figure(figsize=(10, 7))

    clusters = sorted(
        clustered_players["Cluster"].unique()
    )

    for cluster in clusters:

        cluster_indices = (
            clustered_players["Cluster"] == cluster
        )

        plt.scatter(
            pca_data[cluster_indices, 0],
            pca_data[cluster_indices, 1],
            label=f"Cluster {cluster}"
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.title(
        "Player Clusters using PCA"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


def explore_clusters():

    cluster_type, players, features = (
        choose_cluster_type()
    )

    if players is None:

        print("\nReturning to main menu.")

        return

    print()
    print(
        f"You selected: {cluster_type}"
    )

    print()
    print("Available players:", len(players))

    # User chooses number of clusters
    while True:

        try:

            number_of_clusters = int(
                input(
                    "\nHow many clusters would "
                    "you like? "
                )
            )

            if (
                number_of_clusters >= 2
                and
                number_of_clusters < len(players)
            ):

                break

            print(
                "Please enter a number of clusters "
                "between 2 and the number of players."
            )

        except ValueError:

            print(
                "Please enter a whole number."
            )

    # Standardise features
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        players[features]
    )

    # Run k-means clustering
    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        scaled_data
    )

    # Add cluster numbers to DataFrame
    players["Cluster"] = cluster_labels


    show_cluster_statistics(
        players,
        features,
        number_of_clusters
    )


    while True:

        print()
        print("-" * 50)
        print("             CLUSTER OPTIONS")
        print("-" * 50)

        print()
        print("1. Show players in each cluster")
        print("2. Find a player's cluster")
        print("3. View clusters in 2D")
        print("4. Return to main menu")

        choice = input(
            "\nSelect option: "
        ).strip()

        if choice == "1":

            show_cluster_players(
                players,
                number_of_clusters
            )

        elif choice == "2":

            find_player_cluster(
                players
            )

        elif choice == "3":

            show_cluster_pca(
                players,
                features
            )

        elif choice == "4":

            print(
                "\nReturning to main menu."
            )

            return

        else:

            print(
                "\nInvalid option."
            )


# Top players by stat

def show_top_players(df):

    print()
    print("-" * 40)
    print("       TOP PLAYERS BY STATISTIC")
    print("-" * 40)

    feature_name, features = choose_feature_set()

    if features is None:

        print("\nReturning to main menu.")

        return


    # The feature list can be position specific,
    # But the ranking includes ALL players.


    players = df[
        ~df["Pos"].str.contains("GK")
    ].copy()

    print()
    print(
        f"{feature_name} features:"
    )

    for i, feature in enumerate(
        features,
        start=1
    ):

        print(
            f"{i}. {stat_names[feature]}"
        )

    while True:

        try:

            choice = int(
                input(
                    "\nSelect a feature: "
                )
            )

            if 1 <= choice <= len(features):

                break

            print(
                "Please choose a valid feature number."
            )

        except ValueError:

            print(
                "Please enter a number."
            )

    selected_feature = features[
        choice - 1
    ]

    selected_stat_name = stat_names[
        selected_feature
    ]


    top_players = players.sort_values(
        selected_feature,
        ascending=False
    ).head(10)

    print()
    print("-" * 75)
    print(
        f"TOP 10 PLAYERS — {selected_stat_name}"
    )
    print("-" * 75)

    print()

    for i, (_, row) in enumerate(
        top_players.iterrows(),
        start=1
    ):

        print(
            f"{i:>2}. "
            f"{row['Player']:<25} "
            f"{row['Pos']:<7} "
            f"{row['Squad']:<25} "
            f"Age: {row['Age']:<5} "
            f"{selected_stat_name}: "
            f"{row[selected_feature]:.2f}"
        )

    print()
    print("Returning to main menu.")


# Main menu

def show_menu():

    print()
    print("=" * 40)
    print("       FOOTBALL SCOUTING TOOL")
    print("-" * 40)

    print()
    print("1. Compare players")
    print("2. Find similar players")
    print("3. Explore player clusters")
    print("4. Find top players by statistic")
    print("5. Exit")


# Main program loop

while True:

    show_menu()

    choice = input(
        "\nSelect option: "
    ).strip()

    if choice == "1":

        compare_players(df)

    elif choice == "2":

        find_similar_players(df)

    elif choice == "3":

        explore_clusters()

    elif choice == "4":

        show_top_players(df)

    elif choice == "5":

        print("\nGoodbye!")

        break

    else:

        print(
            "\nInvalid option. "
            "Please choose 1-5."
        )
