use std::time::Duration;

use bevy::{prelude::*, sprite::Mesh2dHandle};
use bevy_rapier2d::control::KinematicCharacterController;
use candle_core::Tensor;
use rand::distributions::Distribution;

use crate::{
    gridworld::{LevelLayout, ShouldRun, GRID_CELL_SIZE},
    models::PolicyNet,
    net::{load_weights_into_net, NNWrapper},
    observations::{encode_obs, encode_state, AgentState},
    observer::{Observable, Observer},
    world_objs::NoiseSource,
};

/// Plugin for agent stuff.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            ((
                update_observations,
                move_agents,
                visualize_agent::<PursuerAgent>(Color::RED),
                visualize_agent::<PlayerAgent>(Color::GREEN),
            )
                .run_if(resource_exists::<ShouldRun>),),
        );
    }
}

/// Adds playable functionality for agents.
pub struct AgentPlayPlugin;

impl Plugin for AgentPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                (set_player_action, set_pursuer_action).run_if(resource_exists::<ShouldRun>),
                load_weights_into_net::<PolicyNet>,
            ),
        );
    }
}

/// State used by all agents.
#[derive(Component, Clone, Copy)]
pub struct Agent {
    /// The direction the agent is currently looking at.
    pub dir: Vec2,
}

impl Default for Agent {
    fn default() -> Self {
        Self { dir: Vec2::X }
    }
}

// Indicates the Pursuer agent.
#[derive(Component)]
pub struct PursuerAgent {
    pub observations: Option<(Tensor, Option<Tensor>, Option<Tensor>)>,
    pub obs_timer: Timer,
    pub agent_state: Option<AgentState>,
}

impl Default for PursuerAgent {
    fn default() -> Self {
        Self {
            observations: None,
            obs_timer: Timer::from_seconds(0.5, TimerMode::Repeating),
            agent_state: None,
        }
    }
}

/// Updates the Pursuer's observations.
#[allow(clippy::too_many_arguments)]
fn update_observations(
    mut pursuer_query: Query<&mut PursuerAgent>,
    p_query: Query<(&Agent, &GlobalTransform, &Observer), With<PursuerAgent>>,
    time: Res<Time>,
    observable_query: Query<(Entity, &GlobalTransform), With<Observable>>,
    noise_query: Query<(Entity, &GlobalTransform, &NoiseSource)>,
    level: Res<LevelLayout>,
    player_query: Query<Entity, With<PlayerAgent>>,
    listening_query: Query<(Entity, &GlobalTransform, &NoiseSource)>,
) {
    for mut pursuer in pursuer_query.iter_mut() {
        if let Ok(player_e) = player_query.get_single() {
            pursuer.obs_timer.tick(time.delta());
            if pursuer.obs_timer.just_finished() {
                // Encode observations
                let agent_state = encode_state(&p_query, &listening_query, &level);
                let (grid, _, _) = encode_obs(
                    &observable_query,
                    &noise_query,
                    player_e,
                    &level,
                    &agent_state,
                )
                .unwrap();
                pursuer.observations = Some((grid, None, None));
                pursuer.agent_state = Some(agent_state);
            }
        }
    }
}

/// Indicates the Player agent;
#[derive(Component)]
pub struct PlayerAgent;

/// The child of an `Agent` that contains its visuals.
#[derive(Component)]
pub struct AgentVisuals;

/// Adds a visual to newly created agents.
fn visualize_agent<T: Component>(
    color: Color,
) -> impl Fn(
    Commands<'_, '_>,
    Query<'_, '_, Entity, Added<T>>,
    ResMut<Assets<Mesh>>,
    ResMut<Assets<ColorMaterial>>,
) {
    move |mut commands: Commands,
          agent_query: Query<Entity, Added<T>>,
          mut meshes: ResMut<Assets<Mesh>>,
          mut materials: ResMut<Assets<ColorMaterial>>| {
        for e in agent_query.iter() {
            commands.entity(e).insert((
                Mesh2dHandle(meshes.add(Circle::new(GRID_CELL_SIZE * 0.25))),
                materials.add(color),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
        }
    }
}

const AGENT_SPEED: f32 = GRID_CELL_SIZE * 2.;

/// Holds the next action for an agent.
#[derive(Default, Component)]
pub struct NextAction {
    /// Which direction the agent will move in.
    pub dir: Vec2,
    /// Whether the agent should toggle nearby objects this frame.
    pub toggle_objs: bool,
}

/// Allows the player to set the Player's next action.
fn set_player_action(
    inpt: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<&mut NextAction, With<PlayerAgent>>,
) {
    let mut dir = Vec2::ZERO;
    if inpt.pressed(KeyCode::KeyW) {
        dir.y += 1.;
    }
    if inpt.pressed(KeyCode::KeyS) {
        dir.y -= 1.;
    }
    if inpt.pressed(KeyCode::KeyA) {
        dir.x -= 1.;
    }
    if inpt.pressed(KeyCode::KeyD) {
        dir.x += 1.;
    }
    if let Ok(mut next_action) = player_query.get_single_mut() {
        next_action.dir = dir;
        next_action.toggle_objs = false;
        if inpt.just_pressed(KeyCode::KeyF) {
            next_action.toggle_objs = true;
        }
    }
}

/// Updates the Pursuer's next action.
fn set_pursuer_action(
    net_query: Query<&NNWrapper<PolicyNet>>,
    mut pursuer_query: Query<(&mut NextAction, &PursuerAgent)>,
) {
    if let Ok((mut next_action, pursuer)) = pursuer_query.get_single_mut() {
        let p_net = net_query.single();
        if let Some(net) = &p_net.net {
            if pursuer.obs_timer.just_finished() {
                if let Some((grid, objs, objs_attn_mask)) = &pursuer.observations {
                    let logits = net
                        .forward(
                            &grid.unsqueeze(0).unwrap(),
                            objs.as_ref().map(|t| t.unsqueeze(0).unwrap()).as_ref(),
                            objs_attn_mask
                                .as_ref()
                                .map(|t| t.unsqueeze(0).unwrap())
                                .as_ref(),
                        )
                        .unwrap()
                        .squeeze(0)
                        .unwrap();
                    let probs = (logits
                        .exp()
                        .unwrap()
                        .broadcast_div(&logits.exp().unwrap().sum_all().unwrap()))
                    .unwrap();
                    let index =
                        rand::distributions::WeightedIndex::new(probs.to_vec1::<f32>().unwrap())
                            .unwrap();
                    let mut rng = rand::thread_rng();
                    let action = index.sample(&mut rng);
                    let dir = match action {
                        1 => Vec2::Y,
                        2 => (Vec2::Y + Vec2::X).normalize(),
                        3 => Vec2::X,
                        4 => (-Vec2::Y + Vec2::X).normalize(),
                        5 => -Vec2::Y,
                        6 => (-Vec2::Y + -Vec2::X).normalize(),
                        7 => -Vec2::X,
                        8 => (Vec2::Y + -Vec2::X).normalize(),
                        _ => Vec2::ZERO,
                    };

                    next_action.dir = dir;
                    next_action.toggle_objs = false;
                }
            }
        }
    }
}

/// Moves agents around.
pub fn move_agents(
    mut agent_query: Query<(
        Entity,
        &mut Agent,
        &mut KinematicCharacterController,
        &NextAction,
        &Children,
    )>,
    child_query: Query<(Entity, Option<&Name>, Option<&Children>)>,
    mut vis_query: Query<&mut Transform, With<AgentVisuals>>,
    mut anim_query: Query<&mut AnimationPlayer>,
    time: Res<Time>,
    asset_server: Res<AssetServer>,
) {
    for (agent_e, mut agent, mut controller, next_action, children) in agent_query.iter_mut() {
        let dir = next_action.dir;
        let anim_e = get_entity(&agent_e, &["", "", "Root"], &child_query);
        if let Some(anim_e) = anim_e {
            if let Ok(mut anim) = anim_query.get_mut(anim_e) {
                if dir.length_squared() > 0.1 {
                    let dir = dir.normalize();
                    agent.dir = dir;
                    controller.translation = Some(dir * AGENT_SPEED * time.delta_seconds());
                    for child in children.iter() {
                        if let Ok(mut xform) = vis_query.get_mut(*child) {
                            xform.look_to(-dir.extend(0.), Vec3::Z);
                            {
                                anim.play_with_transition(
                                    asset_server.load("characters/cyborgFemaleA.glb#Animation1"),
                                    Duration::from_secs_f32(0.2),
                                )
                                .repeat();
                            }
                            break;
                        }
                    }
                } else {
                    anim.play_with_transition(
                        asset_server.load("characters/cyborgFemaleA.glb#Animation0"),
                        Duration::from_secs_f32(0.2),
                    )
                    .repeat();
                }
            }
        }
    }
}

/// Returns the entity at this path.
fn get_entity(
    parent: &Entity,
    path: &[&str],
    child_query: &Query<(Entity, Option<&Name>, Option<&Children>)>,
) -> Option<Entity> {
    if path.is_empty() {
        return Some(*parent);
    }
    if let Ok((_, _, Some(children))) = child_query.get(*parent) {
        for child in children {
            let (_, name, _) = child_query.get(*child).unwrap();
            if (name.is_none() && path[0].is_empty()) || (name.unwrap().as_str() == path[0]) {
                let e = get_entity(child, &path[1..], child_query);
                if e.is_some() {
                    return e;
                }
            }
        }
    }
    None
}
