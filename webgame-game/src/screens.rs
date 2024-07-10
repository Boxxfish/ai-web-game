use bevy::{
    prelude::*,
    ui::{UiBatch, UiImageBindGroups},
};

use crate::{
    gridworld::{LevelLoader, GRID_CELL_SIZE},
    ui::{
        menu_button::{MenuButton, MenuButtonBundle, MenuButtonPressedEvent},
        screen_transition::{FadeFinishedEvent, ScreenTransitionBundle, StartFadeEvent},
    },
};

/// Describes and handles logic for various screens.
pub struct ScreensPlayPlugin;

impl Plugin for ScreensPlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_state(ScreenState::TitleScreen)
            .add_systems(Startup, init_ui)
            .add_systems(OnEnter(ScreenState::TitleScreen), init_title_screen)
            .add_systems(OnExit(ScreenState::TitleScreen), destroy_title_screen)
            .add_systems(
                Update,
                (handle_title_screen_transition, handle_title_screen_btns)
                    .run_if(in_state(ScreenState::TitleScreen)),
            )
            .add_systems(OnEnter(ScreenState::LevelSelect), init_level_select)
            .add_systems(OnExit(ScreenState::LevelSelect), destroy_level_select)
            .add_systems(
                Update,
                (handle_level_select_transition, handle_level_select_btns)
                    .run_if(in_state(ScreenState::LevelSelect)),
            )
            .add_systems(OnEnter(ScreenState::Game), init_game)
            .add_systems(OnExit(ScreenState::Game), destroy_game)
            .add_systems(OnEnter(ScreenState::About), init_about)
            .add_systems(OnExit(ScreenState::About), destroy_about)
            .add_systems(
                Update,
                (handle_about_transition, handle_about_btns).run_if(in_state(ScreenState::About)),
            );
    }
}

/// The screens we can be on.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, States)]
pub enum ScreenState {
    TitleScreen,
    LevelSelect,
    Game,
    About,
}

/// Denotes the title screen.
#[derive(Component)]
struct TitleScreen;

/// Actions that can be performed on the title screen.
#[derive(Component, Copy, Clone)]
enum TitleScreenAction {
    Start,
    About,
}

/// Holds the state to transition to when the transition finishes.
#[derive(Resource)]
struct TransitionNextState<T>(pub T);

/// Initializes UI elements that persist across scenes.
fn init_ui(mut commands: Commands) {
    commands.spawn(ScreenTransitionBundle::default());
}

fn init_title_screen(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
) {
    let font_bold = asset_server.load("fonts/montserrat/Montserrat-Bold.ttf");
    ev_start_fade.send(StartFadeEvent { fade_in: true });

    commands.spawn((Camera2dBundle::default(), IsDefaultUiCamera));
    commands
        .spawn((
            TitleScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Background
            p.spawn(ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    margin: UiRect::all(Val::Auto),
                    ..default()
                },
                image: asset_server.load("ui/title_screen/background.png").into(),
                z_index: ZIndex::Local(0),
                ..default()
            });

            // Text elements
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.with_a(0.9).into(),
                z_index: ZIndex::Local(1),
                ..default()
            })
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
                        width: Val::Px(128.),
                        height: Val::Percent(50.),
                        margin: UiRect::horizontal(Val::Auto),
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Start,
                        padding: UiRect::vertical(Val::Px(16.)),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    for (action, label) in [
                        (TitleScreenAction::Start, "START"),
                        (TitleScreenAction::About, "ABOUT"),
                    ] {
                        p.spawn((action, MenuButtonBundle::from_label(label)));
                    }
                });
            });
        });
}

fn destroy_title_screen(
    mut commands: Commands,
    screen_query: Query<Entity, With<TitleScreen>>,
    cam_query: Query<Entity, With<Camera2d>>,
) {
    commands.entity(screen_query.single()).despawn_recursive();
    commands.entity(cam_query.single()).despawn_recursive();
}

fn handle_title_screen_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    action_query: Query<&TitleScreenAction>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    mut commands: Commands,
) {
    for ev in ev_btn_pressed.read() {
        if let Ok(action) = action_query.get(ev.sender) {
            ev_start_fade.send(StartFadeEvent { fade_in: false });
            commands.insert_resource(TransitionNextState(*action));
        }
    }
}

fn handle_title_screen_transition(
    mut ev_fade_finished: EventReader<FadeFinishedEvent>,
    mut commands: Commands,
    transition_state: Option<Res<TransitionNextState<TitleScreenAction>>>,
    mut next_state: ResMut<NextState<ScreenState>>,
    screen_query: Query<Entity, With<TitleScreen>>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in && !screen_query.is_empty() {
            commands.remove_resource::<TransitionNextState<TitleScreenAction>>();
            next_state.0 = match transition_state.as_ref().unwrap().0 {
                TitleScreenAction::Start => Some(ScreenState::LevelSelect),
                TitleScreenAction::About => Some(ScreenState::About),
            }
        }
    }
}

fn init_game(mut commands: Commands, mut ev_start_fade: EventWriter<StartFadeEvent>) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(
                GRID_CELL_SIZE * (((8 + 1) / 2) as f32),
                -300.,
                700.,
            ))
            .with_rotation(Quat::from_rotation_x(0.5)),
            projection: Projection::Perspective(PerspectiveProjection {
                fov: 0.4,
                ..default()
            }),
            ..default()
        },
        IsDefaultUiCamera,
    ));

    ev_start_fade.send(StartFadeEvent { fade_in: true });
}

fn destroy_game() {}

/// Denotes the level select screen.
#[derive(Component)]
struct LevelSelectScreen;

/// A button that loads a level.
#[derive(Component)]
struct LevelSelectButton {
    pub level: String,
}

#[derive(Component, Clone)]
enum LevelSelectAction {
    Level(String),
    Back,
}

fn init_level_select(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
) {
    ev_start_fade.send(StartFadeEvent { fade_in: true });

    commands.spawn((Camera2dBundle::default(), IsDefaultUiCamera));
    commands
        .spawn((
            LevelSelectScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Background
            p.spawn(ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    margin: UiRect::all(Val::Auto),
                    ..default()
                },
                image: asset_server.load("ui/title_screen/background.png").into(),
                z_index: ZIndex::Local(0),
                ..default()
            });

            // Main elements
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.with_a(0.95).into(),
                z_index: ZIndex::Local(1),
                ..default()
            })
            .with_children(|p| {
                p.spawn(NodeBundle {
                    style: Style {
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        column_gap: Val::Px(16.),
                        row_gap: Val::Px(16.),
                        margin: UiRect::axes(Val::Auto, Val::Px(8.)),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    for level in ["test.json", "test.json", "test.json"] {
                        // let image = asset_server.load(format!("ui/level_images/{level}.png"));
                        p.spawn((
                            LevelSelectButton {
                                level: format!("levels/{level}.json"),
                            },
                            MenuButtonBundle::from_label(level),
                        ));
                    }
                });

                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(128.),
                        margin: UiRect::axes(Val::Auto, Val::Auto),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    p.spawn(MenuButtonBundle::from_label("Back"));
                });
            });
        });
}

fn destroy_level_select(
    mut commands: Commands,
    screen_query: Query<Entity, With<LevelSelectScreen>>,
    cam_query: Query<Entity, With<Camera2d>>,
) {
    commands.entity(screen_query.single()).despawn_recursive();
    commands.entity(cam_query.single()).despawn_recursive();
}

fn handle_level_select_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    action_query: Query<&LevelSelectAction>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    mut commands: Commands,
) {
    for ev in ev_btn_pressed.read() {
        if let Ok(action) = action_query.get(ev.sender) {
            ev_start_fade.send(StartFadeEvent { fade_in: false });
            commands.insert_resource(TransitionNextState(action.clone()));
        }
    }
}

fn handle_level_select_transition(
    mut ev_fade_finished: EventReader<FadeFinishedEvent>,
    mut commands: Commands,
    transition_state: Option<Res<TransitionNextState<LevelSelectAction>>>,
    mut next_state: ResMut<NextState<ScreenState>>,
    screen_query: Query<Entity, With<TitleScreen>>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in && !screen_query.is_empty() {
            commands.remove_resource::<TransitionNextState<LevelSelectAction>>();
            next_state.0 = match &transition_state.as_ref().unwrap().0 {
                LevelSelectAction::Level(level) => {
                    commands.insert_resource(LevelLoader::Path(level.clone()));
                    Some(ScreenState::Game)
                }
                LevelSelectAction::Back => Some(ScreenState::TitleScreen),
            }
        }
    }
}

/// Denotes the about screen.
#[derive(Component)]
struct AboutScreen;

fn init_about(
    mut commands: Commands,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    asset_server: Res<AssetServer>,
) {
    ev_start_fade.send(StartFadeEvent { fade_in: true });

    commands.spawn((Camera2dBundle::default(), IsDefaultUiCamera));
    commands
        .spawn((
            AboutScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Background
            p.spawn(ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    margin: UiRect::all(Val::Auto),
                    ..default()
                },
                image: asset_server.load("ui/title_screen/background.png").into(),
                z_index: ZIndex::Local(0),
                ..default()
            });

            // Text elements
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.with_a(0.95).into(),
                z_index: ZIndex::Local(1),
                ..default()
            })
            .with_children(|p| {
                p.spawn(NodeBundle {
                    style: Style {
                        flex_direction: FlexDirection::Column,
                        display: Display::Flex,
                        padding: UiRect::axes(Val::Px(32.), Val::Px(16.)),
                        ..default()
                    },
                    z_index: ZIndex::Local(1),
                    ..default()
                }).with_children(|p| {
                let font_regular = asset_server.load("fonts/montserrat/Montserrat-Regular.ttf");
                let color = Color::WHITE;
                let heading = TextStyle {
                    font: font_regular.clone(),
                    font_size: 28.,
                    color,
                };
                let regular = TextStyle {
                    font: font_regular.clone(),
                    font_size: 16.,
                    color,
                };
                p.spawn(TextBundle::from_sections([
                    TextSection::new("About\n", heading.clone()),
                    TextSection::new("\nThis game demonstrates how machine learning can be used to create intelligent pursuer-type enemies.\n\n", regular.clone()),
                    TextSection::new(
                        "The pursuer is equipped with a discrete Bayes filter, allowing it to use evidence from its environment to track you down. It does this by generating a map of where it thinks you are and continuously updating it based on data from its sensors. You can toggle this map in-game to see where the pursuer thinks you are.\n\n", 
                        regular.clone()
                    ),
                    TextSection::new(
                        "The pursuer has also been trained to chase after you, using reinforcement learning.\n\n",
                        regular.clone()
                    ),
                    TextSection::new(
                        "Instructions\n",
                        heading.clone()
                    ),
                    TextSection::new(
                        "\nFind the key in each level and use it to escape. Be careful, though — there's someone locked in with you roaming the halls, and you REALLY don't want to come face to face with them!\n\n",
                        regular.clone()
                    ),
                ]));
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(128.),
                        margin: UiRect::axes(Val::Auto, Val::Auto),
                        ..default()
                    },
                    ..default()
                }).with_children(|p| {
                    p.spawn(MenuButtonBundle::from_label("Back"));
                });
            });
        });
    });
}

fn destroy_about(
    mut commands: Commands,
    screen_query: Query<Entity, With<AboutScreen>>,
    cam_query: Query<Entity, With<Camera2d>>,
) {
    commands.entity(screen_query.single()).despawn_recursive();
    commands.entity(cam_query.single()).despawn_recursive();
}

fn handle_about_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
) {
    for _ in ev_btn_pressed.read() {
        ev_start_fade.send(StartFadeEvent { fade_in: false });
    }
}

fn handle_about_transition(
    mut ev_fade_finished: EventReader<FadeFinishedEvent>,
    mut next_state: ResMut<NextState<ScreenState>>,
    screen_query: Query<Entity, With<AboutScreen>>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in && !screen_query.is_empty() {
            next_state.0 = Some(ScreenState::TitleScreen);
        }
    }
}
