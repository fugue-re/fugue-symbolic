use fugue::bytes::Order;
use fugue::ir::il::Location;

use thiserror::Error;

use crate::builder::{SymExpr, SymExprBuilderMut};
use crate::value::ConcolicValue;

#[derive(Debug, Error)]
pub enum Error {
    #[error("cannot solve constraints to concretise value")]
    CannotSolve,
}

#[derive(Debug, Clone)]
pub struct PathConstraint {
    source: Location,
    target: Location,
    condition: bool,
    constraint: SymExpr,
}

impl PathConstraint {
    pub fn new(
        source: Location,
        target: Location,
        condition: bool,
        constraint: SymExpr,
    ) -> Self {
        Self {
            source,
            target,
            condition,
            constraint,
        }
    }

    pub fn condition(&self) -> bool {
        self.condition
    }

    pub fn constraint(&self) -> &SymExpr {
        &self.constraint
    }

    pub fn source(&self) -> &Location {
        &self.source
    }

    pub fn target(&self) -> &Location {
        &self.target
    }
}

#[derive(Debug, Clone)]
pub struct PathConstraints {
    true_condition: PathConstraint,
    false_condition: PathConstraint,
    taken_location: Location,
    taken_condition: bool,
}

impl PathConstraints {
    pub fn new(
        tcondition: PathConstraint,
        fcondition: PathConstraint,
        taken_location: Location,
        taken_condition: bool,
    ) -> Self {
        Self {
            true_condition: tcondition,
            false_condition: fcondition,
            taken_location,
            taken_condition,
        }
    }

    pub fn taken(&self) -> &PathConstraint {
        if self.taken_condition {
            &self.true_condition
        } else {
            &self.false_condition
        }
    }

    pub fn not_taken(&self) -> &PathConstraint {
        if self.taken_condition {
            &self.false_condition
        } else {
            &self.true_condition
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PathManager {
    constraints: Vec<PathConstraints>,
    strategy: (), // I guess we'll add something here...
}

impl PathManager {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn push_constraint<O: Order>(
        &mut self,
        mut builder: SymExprBuilderMut,
        condition: ConcolicValue<bool>,
        source: Location,
        tlocation: Location,
        flocation: Location,
    ) -> Result<bool, Error> {
        let f = builder.constant(false);
        let cv = condition
            .clone()
            .unwrap_concolic(
                |v| builder.constant(v),
                |v| v
            );

        let tv = builder.not_equal(cv.clone(), f.clone());
        let fv = builder.equal(cv, f);

        // For now, we apply a concretise always strategy; in the future,
        // we probably want to force following a fixed path, etc.
        let condition_val = condition.try_unwrap_concolic(
            |value| Ok(value),
            |expr| builder.solve_value::<bool, O>(expr, &self).map_err(|_| Error::CannotSolve)
        )?;

        let location = if condition_val {
            tlocation.clone()
        } else {
            flocation.clone()
        };

        let pcs = PathConstraints::new(
            PathConstraint::new(source.clone(), tlocation, true, tv),
            PathConstraint::new(source, flocation, false, fv),
            location,
            condition_val,
        );

        self.constraints.push(pcs);

        Ok(condition_val)
    }

    pub fn pop_constraint(&mut self) -> Option<PathConstraints> {
        self.constraints.pop()
    }

    pub fn path_constraints(&self) -> impl Iterator<Item=&PathConstraints> {
        self.constraints.iter()
    }
}
