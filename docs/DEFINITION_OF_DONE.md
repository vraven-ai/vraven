# VRAVEN Definition of Done

A feature is complete only when all relevant conditions are met.

## Engineering

* Implementation is typed where practical
* No unnecessary tensor copies exist
* Hooks are removed after use
* CPU behaviour is tested
* CUDA and MPS tests are availability guarded
* Failure modes are explicit
* Model parameters remain unchanged unless causal mode documents an intervention

## Scientific

* The metric has a mathematical definition
* Interpretation is documented
* Assumptions are documented
* Limitations are documented
* At least one numerical correctness test exists
* Validation method is specified
* Observational and causal evidence are clearly separated

## Performance

* Timing impact has been measured for performance sensitive changes
* Memory impact has been measured where tensor capture changes
* No regression is accepted without documented justification

## Documentation

* Public behaviour is documented
* An example is added when user facing
* Terminology is consistent with VRAVEN documents
* Agent facing implementation rules are updated when architecture changes

## Visualisation

A visualisation is complete only when:

* It answers one explicit scientific question
* Every visual variable maps to a documented measured quantity
* It returns an independent figure
* `interactive_dark` is implemented
* `publication_light` is implemented
* Both modes use identical scientific values and semantic meanings
* Publication mode has a white background and no glow
* Interactive mode uses dark styling and meaningful interaction
* VRAVEN theme functions are used instead of library defaults
* Values are traceable to the input trace and metric version
* A legend or direct labels are present where needed
* Colour is not the only critical encoding
* Projection seeds, alignment and normalisation are recorded where applicable
* Labels are not clipped at target export sizes
* A vector export path exists where the plotting technology supports it
* Tests protect the most important mapping and both render modes
* Limitations are stored in documentation or figure metadata
