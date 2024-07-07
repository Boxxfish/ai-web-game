use bevy::prelude::*;

use crate::gridworld::LevelLoader;

/// Describes and handles logic for various screens.
pub struct ScreensPlayPlugin;

impl Plugin for ScreensPlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_state(ScreenState::TitleScreen)
            .add_systems(OnEnter(ScreenState::TitleScreen), init_title_screen)
            .add_systems(OnExit(ScreenState::TitleScreen), destroy_title_screen)
            .add_systems(OnEnter(ScreenState::Game), init_game)
            .add_systems(OnExit(ScreenState::Game), destroy_game);
    }
}

/// The screens we can be on.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, States)]
pub enum ScreenState {
    TitleScreen,
    Game,
}

/// Denotes the title screen.
#[derive(Component)]
struct TitleScreen;

fn init_title_screen(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font_bold = asset_server.load("fonts/montserrat/Montserrat-Bold.ttf");
    let font_regular = asset_server.load("fonts/montserrat/Montserrat-Regular.ttf");

    commands.spawn((Camera2dBundle::default(), IsDefaultUiCamera));
    commands
        .spawn((
            TitleScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Top text
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(50.),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::End,
                    ..default()
                },
                ..default()
            })
            .with_children(|p| {
                p.spawn(TextBundle::from_section(
                    "DEMO:",
                    TextStyle {
                        font: font_bold.clone(),
                        font_size: 40.,
                        color: Color::WHITE,
                    },
                ));
                p.spawn(TextBundle::from_section(
                    "PURSUER",
                    TextStyle {
                        font: font_bold.clone(),
                        font_size: 64.,
                        color: Color::WHITE,
                    },
                ));
            });
            // Options
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(50.),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Start,
                    padding: UiRect::vertical(Val::Px(32.)),
                    ..default()
                },
                ..default()
            })
            .with_children(|p| {
                p.spawn(TextBundle::from_section(
                    "START",
                    TextStyle {
                        font: font_regular.clone(),
                        font_size: 22.,
                        color: Color::WHITE,
                    },
                ));
                p.spawn(TextBundle::from_section(
                    "ABOUT",
                    TextStyle {
                        font: font_regular.clone(),
                        font_size: 22.,
                        color: Color::WHITE,
                    },
                ));
            });
        });
    // });
}

fn destroy_title_screen() {}

fn init_game(mut commands: Commands) {
    commands.insert_resource(LevelLoader::Path("levels/test.json".into()));
}

fn destroy_game() {}
