use std::time::Duration;

use bevy::{prelude::*, render::extract_resource::ExtractResource};

use crate::{
    gridworld::LevelLoader,
    ui::menu_button::{MenuButton, MenuButtonBundle, MenuButtonPressedEvent},
};

/// Describes and handles logic for various screens.
pub struct ScreensPlayPlugin;

impl Plugin for ScreensPlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_state(ScreenState::TitleScreen)
            .add_event::<StartFadeEvent>()
            .add_event::<FadeFinishedEvent>()
            .add_systems(OnEnter(ScreenState::TitleScreen), init_title_screen)
            .add_systems(OnExit(ScreenState::TitleScreen), destroy_title_screen)
            .add_systems(OnEnter(ScreenState::Game), init_game)
            .add_systems(OnExit(ScreenState::Game), destroy_game)
            .add_systems(
                Update,
                (
                    handle_title_screen_transition,
                    handle_title_screen_btns,
                    handle_fade_transition,
                    handle_fade_evs,
                ),
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

fn init_title_screen(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font_bold = asset_server.load("fonts/montserrat/Montserrat-Bold.ttf");

    commands.spawn((Camera2dBundle::default(), IsDefaultUiCamera));
    commands.spawn((
        ScreenTransition::default(),
        NodeBundle {
            style: Style {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                position_type: PositionType::Absolute,
                ..default()
            },
            background_color: Color::BLACK.into(),
            z_index: ZIndex::Global(100),
            ..default()
        },
    ));
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

/// Holds the state to transition to when the transition finishes.
#[derive(Resource)]
struct TransitionNextState(pub TitleScreenAction);

/// Denotes the screen transition.
#[derive(Component)]
pub struct ScreenTransition {
    pub fade_in: bool,
    pub finished: bool,
    pub alpha_amount: f32,
}

impl Default for ScreenTransition {
    fn default() -> Self {
        Self {
            fade_in: true,
            finished: false,
            alpha_amount: 1.,
        }
    }
}

/// Sent when the transition should be run.
#[derive(Event)]
pub struct StartFadeEvent {
    pub fade_in: bool,
}

/// Sent when the transition finished.
#[derive(Event)]
pub struct FadeFinishedEvent {
    pub fade_in: bool,
}

/// Responds to fade events.
fn handle_fade_evs(
    mut transition_query: Query<&mut ScreenTransition>,
    mut ev_start_fade: EventReader<StartFadeEvent>,
) {
    for ev in ev_start_fade.read() {
        for mut transition in transition_query.iter_mut() {
            transition.fade_in = ev.fade_in;
            transition.finished = false;
        }
    }
}

const TRANSITION_SECS: f32 = 0.5;
const MIN_TRANSITION: f32 = 0.001;

/// Updates the screen transition.
fn handle_fade_transition(
    mut transition_query: Query<(&mut ScreenTransition, &mut BackgroundColor)>,
    time: Res<Time>,
    mut ev_fade_finished: EventWriter<FadeFinishedEvent>,
) {
    for (mut transition, mut bg_color) in transition_query.iter_mut() {
        if !transition.finished {
            let delta = (1. / TRANSITION_SECS) * time.delta_seconds();
            if transition.fade_in {
                transition.alpha_amount = f32::max(transition.alpha_amount - delta, 0.);
                if transition.alpha_amount < MIN_TRANSITION {
                    transition.finished = true;
                }
            } else {
                transition.alpha_amount = f32::min(transition.alpha_amount + delta, 1.);
                if transition.alpha_amount > (1. - MIN_TRANSITION) {
                    transition.finished = true;
                }
            }
            bg_color.0.set_a(transition.alpha_amount);
            if transition.finished {
                ev_fade_finished.send(FadeFinishedEvent {
                    fade_in: transition.fade_in,
                });
            }
        }
    }
}
