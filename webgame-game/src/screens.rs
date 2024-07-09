use bevy::prelude::*;

use crate::{
    gridworld::LevelLoader,
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
            .add_systems(OnEnter(ScreenState::Game), init_game)
            .add_systems(OnExit(ScreenState::Game), destroy_game)
            .add_systems(
                Update,
                (handle_title_screen_transition, handle_title_screen_btns),
            );
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

/// Actions that can be performed on the title screen.
#[derive(Component, Copy, Clone)]
enum TitleScreenAction {
    Start,
    About,
}

/// Holds the state to transition to when the transition finishes.
#[derive(Resource)]
struct TransitionNextState(pub TitleScreenAction);

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
    transition_state: Option<Res<TransitionNextState>>,
    mut next_state: ResMut<NextState<ScreenState>>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in {
            commands.remove_resource::<TransitionNextState>();
            next_state.0 = match transition_state.as_ref().unwrap().0 {
                TitleScreenAction::Start => Some(ScreenState::Game),
                TitleScreenAction::About => Some(ScreenState::Game),
            }
        }
    }
}

fn init_game(mut commands: Commands, mut ev_start_fade: EventWriter<StartFadeEvent>) {
    commands.insert_resource(LevelLoader::Path("levels/test.json".into()));
    ev_start_fade.send(StartFadeEvent { fade_in: true });
}

fn destroy_game() {}
