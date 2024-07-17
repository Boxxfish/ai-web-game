use std::marker::PhantomData;

use bevy::prelude::*;

#[derive(Default)]
pub struct ScreenTransitionPlugin<T: States> {
    t: PhantomData<T>,
}

impl<T: States> Plugin for ScreenTransitionPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_event::<StartFadeEvent>()
            .add_event::<FadeFinishedEvent<T>>()
            .add_systems(Update, (handle_fade_transition::<T>, handle_fade_evs));
    }
}

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

/// A bundle for creating `ScreenTransition`s.
#[derive(Bundle)]
pub struct ScreenTransitionBundle {
    pub screen_transition: ScreenTransition,
    pub node_bundle: NodeBundle,
}

impl Default for ScreenTransitionBundle {
    fn default() -> Self {
        Self {
            node_bundle: NodeBundle {
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
            screen_transition: ScreenTransition::default(),
        }
    }
}

/// Sent when the transition should be run.
#[derive(Event)]
pub struct StartFadeEvent {
    pub fade_in: bool,
}

/// Sent when the transition finished.
/// This also sends the current state when this was sent.
#[derive(Event)]
pub struct FadeFinishedEvent<T: States> {
    pub from_state: T,
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

const TRANSITION_SECS: f32 = 0.3;
const MIN_TRANSITION: f32 = 0.001;

/// Updates the screen transition.
fn handle_fade_transition<T: States>(
    mut transition_query: Query<(&mut ScreenTransition, &mut BackgroundColor)>,
    time: Res<Time>,
    mut ev_fade_finished: EventWriter<FadeFinishedEvent<T>>,
    state: Res<State<T>>,
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
                    from_state: state.get().to_owned(),
                    fade_in: transition.fade_in,
                });
            }
        }
    }
}
