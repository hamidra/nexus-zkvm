use ark_crypto_primitives::sponge::{
    constraints::{CryptographicSpongeVar, SpongeWithGadget},
    Absorb,
};
use ark_ec::{
    short_weierstrass::{Projective, SWCurveConfig},
    CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_spartan::{
    committed_relaxed_snark as spartan_snark, committed_relaxed_snark::CRSNARKKey as SNARKGens,
    crr1csproof::CRR1CSShape, polycommitments::PolyCommitmentScheme, ComputationCommitment,
    ComputationDecommitment,
};
use ark_std::{cmp::max, marker::PhantomData};
use merlin::Transcript;

use super::{augmented::SQUEEZE_NATIVE_ELEMENTS_NUM, IVCProof, IVCProofNonBase, PublicParams};
use crate::{
    absorb::CryptographicSpongeExt,
    commitment::CommitmentScheme,
    folding::nova::cyclefold::{
        nimfs::{NIMFSProof, R1CSInstance, RelaxedR1CSInstance, RelaxedR1CSWitness},
        Error as NovaError,
    },
    nova::pcd::compression::{
        commitment_utils::PolyVectorCommitment,
        error::{ProofError, SpartanError},
    },
    r1cs::R1CSShape,
    StepCircuit, LOG_TARGET,
};

pub type PVC<G, PC> = PolyVectorCommitment<Projective<G>, PC>;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct CompressedIVCProof<G1, G2, PC, C2, RO, SC>
where
    G1: SWCurveConfig,
    G2: SWCurveConfig,
    C2: CommitmentScheme<Projective<G2>>,
    PC: PolyCommitmentScheme<Projective<G1>>,
    PC::Commitment: Into<Projective<G1>> + From<Projective<G1>> + Copy,
    RO: SpongeWithGadget<G1::ScalarField> + Send + Sync,
    SC: StepCircuit<G1::ScalarField>,
{
    z_0: Vec<G1::ScalarField>,
    pub i: u64,
    pub z_i: Vec<G1::ScalarField>,
    pub U: RelaxedR1CSInstance<G1, PVC<G1, PC>>,
    pub u: R1CSInstance<G1, PVC<G1, PC>>,
    pub U_secondary: RelaxedR1CSInstance<G2, C2>,
    pub W_secondary_prime: RelaxedR1CSWitness<G2>,

    pub spartan_proof: spartan_snark::SNARK<Projective<G1>, PC>,
    pub folding_proof: NIMFSProof<G1, G2, PVC<G1, PC>, C2, RO>,

    _random_oracle: PhantomData<RO>,
    _step_circuit: PhantomData<SC>,
}

#[derive(CanonicalDeserialize, CanonicalSerialize)]
pub struct SNARKKey<G: CurveGroup, PC: PolyCommitmentScheme<G>> {
    shape: CRR1CSShape<G::ScalarField>,
    computation_comm: ComputationCommitment<G, PC>,
    computation_decomm: ComputationDecommitment<G::ScalarField>,
    snark_gens: SNARKGens<G, PC>,
}

impl<G: CurveGroup, PC: PolyCommitmentScheme<G>> SNARKKey<G, PC> {
    /// convenience function to derive the minimum log size of the SRS
    /// needed to support compression for a given `shape`.
    pub fn get_min_srs_size(shape: &R1CSShape<G>) -> usize {
        let R1CSShape {
            num_constraints,
            num_vars,
            num_io,
            A,
            B,
            C,
        } = shape;
        // spartan uses the convention that num_inputs does not include the leading `u`.
        let num_inputs = num_io - 1;
        let num_nz_entries = max(A.len(), max(B.len(), C.len()));
        SNARKGens::<G, PC>::get_min_num_vars(
            *num_constraints,
            *num_vars,
            num_inputs,
            num_nz_entries,
        )
    }
}
pub struct SNARK<G1, G2, PC, C2, RO, SC>
where
    G1: SWCurveConfig,
    G1::BaseField: PrimeField + Absorb,
    G2: SWCurveConfig,
    C2: CommitmentScheme<Projective<G2>>,
    PC: PolyCommitmentScheme<Projective<G1>>,
    RO: SpongeWithGadget<G1::ScalarField> + Send + Sync,
    SC: StepCircuit<G1::ScalarField>,
{
    _group: PhantomData<G1>,
    _group_secondary: PhantomData<G2>,
    _poly_commitment: PhantomData<PC>,
    _commitment: PhantomData<C2>,
    _random_oracle: PhantomData<RO>,
    _step_circuit: PhantomData<SC>,
}

impl<G1, G2, PC, C2, RO, SC> SNARK<G1, G2, PC, C2, RO, SC>
where
    G1: SWCurveConfig,
    G2: SWCurveConfig<BaseField = G1::ScalarField, ScalarField = G1::BaseField>,
    G1::BaseField: PrimeField + Absorb,
    G2::BaseField: PrimeField + Absorb,
    C2: CommitmentScheme<Projective<G2>>,
    PC: PolyCommitmentScheme<Projective<G1>>,
    PC::Commitment: Copy + Into<Projective<G1>> + From<Projective<G1>>,
    RO: SpongeWithGadget<G1::ScalarField> + Send + Sync,
    RO::Config: CanonicalSerialize + CanonicalDeserialize + Sync,
    RO::Var: CryptographicSpongeVar<G1::ScalarField, RO, Parameters = RO::Config>,
    SC: StepCircuit<G1::ScalarField>,
{
    pub fn setup(
        pp: &PublicParams<G1, G2, PVC<G1, PC>, C2, RO, SC>,
        srs: &PC::SRS,
    ) -> Result<SNARKKey<Projective<G1>, PC>, SpartanError> {
        let _span = tracing::debug_span!(target: LOG_TARGET, "Spartan_setup").entered();
        let PublicParams { shape: _shape, .. } = pp;
        // converts the R1CSShape from this crate into a CRR1CSShape from the Spartan crate
        let shape: CRR1CSShape<G1::ScalarField> = _shape.clone().try_into()?;
        let (num_cons, num_vars, num_inputs) = (
            shape.get_num_cons(),
            shape.get_num_vars(),
            shape.get_num_inputs(),
        );

        let num_nz_entries = max(_shape.A.len(), max(_shape.B.len(), _shape.C.len()));
        let snark_gens = SNARKGens::new(srs, num_cons, num_vars, num_inputs, num_nz_entries);
        let (computation_comm, computation_decomm) =
            spartan_snark::SNARK::<Projective<G1>, PC>::encode(&shape.inst, &snark_gens);
        Ok(SNARKKey {
            shape,
            computation_comm,
            computation_decomm,
            snark_gens,
        })
    }

    pub fn compress(
        params: &PublicParams<G1, G2, PVC<G1, PC>, C2, RO, SC>,
        key: &SNARKKey<Projective<G1>, PC>,
        ivc_proof: IVCProof<G1, G2, PVC<G1, PC>, C2, RO, SC>,
    ) -> Result<CompressedIVCProof<G1, G2, PC, C2, RO, SC>, SpartanError> {
        let _span = tracing::debug_span!(target: LOG_TARGET, "Spartan_prove").entered();
        let SNARKKey {
            shape,
            computation_comm,
            computation_decomm,
            snark_gens,
        } = key;
        let IVCProof { z_0, non_base, .. } = ivc_proof;

        let IVCProofNonBase {
            U,
            W,
            U_secondary,
            W_secondary,
            u,
            w,
            i,
            z_i,
        } = non_base.ok_or(SpartanError::InvalidProof(ProofError::InvalidProof))?;

        // First, we fold the instance-witness pair `(u,w)` into the running instances.
        let (folding_proof, (U_prime, W_prime), (_U_secondary_prime, W_secondary_prime)) =
            NIMFSProof::prove(
                &params.pp,
                &params.pp_secondary,
                &params.ro_config,
                &params.digest,
                (&params.shape, &params.shape_secondary),
                (&U, &W),
                (&U_secondary, &W_secondary),
                (&u, &w),
            )?;
        let mut transcript = Transcript::new(b"spartan_snark");
        // Now, we use Spartan to prove knowledge of the witness `W_prime`
        // for the committed relaxed r1cs instance `U_prime`
        let spartan_proof = spartan_snark::SNARK::<Projective<G1>, PC>::prove(
            shape,
            &U_prime.try_into()?,
            W_prime.try_into()?,
            computation_comm,
            computation_decomm,
            snark_gens,
            &mut transcript,
        );

        Ok(CompressedIVCProof {
            z_0: z_0.clone(),
            i,
            z_i: z_i.clone(),
            U,
            u,
            U_secondary,
            W_secondary_prime,
            spartan_proof,
            folding_proof,
            _random_oracle: PhantomData,
            _step_circuit: PhantomData,
        })
    }

    pub fn verify(
        key: &SNARKKey<Projective<G1>, PC>,
        params: &PublicParams<G1, G2, PVC<G1, PC>, C2, RO, SC>,
        proof: &CompressedIVCProof<G1, G2, PC, C2, RO, SC>,
    ) -> Result<(), SpartanError> {
        let _span =
            tracing::debug_span!(target: LOG_TARGET, "Spartan_verify", i = proof.i).entered();
        let CompressedIVCProof {
            z_0,
            i,
            z_i,
            U,
            u,
            U_secondary,
            W_secondary_prime,
            spartan_proof,
            folding_proof,
            ..
        } = proof;

        // First, we hash the running instances U, U_secondary and check that
        // the public IO of `u` is equal to this hash value.
        let mut random_oracle = RO::new(&params.ro_config);
        random_oracle.absorb(&params.digest);
        random_oracle.absorb(&G1::ScalarField::from(*i));
        random_oracle.absorb(z_0);
        random_oracle.absorb(z_i);
        random_oracle.absorb(&U);
        random_oracle.absorb_non_native(&U_secondary);

        let hash: &G1::ScalarField =
            &random_oracle.squeeze_field_elements(SQUEEZE_NATIVE_ELEMENTS_NUM)[0];
        if hash != &u.X[1] {
            return Err(SpartanError::InvalidProof(ProofError::InvalidPublicInput));
        }

        // Now, using the folding proof provided by the prover, we compute the folded
        // instances U_prime and U_secondary_prime.
        let (U_prime, U_secondary_prime) = NIMFSProof::verify(
            folding_proof,
            &params.ro_config,
            &params.digest,
            U,
            U_secondary,
            u,
        )?;

        // We check that the provided witness `W_secondary_prime` satisfies the
        // committed relaxed r1cs instance `U_secondary_prime`.`
        params
            .shape_secondary
            .is_relaxed_satisfied(&U_secondary_prime, W_secondary_prime, &params.pp_secondary)
            .map_err(|_| SpartanError::InvalidProof(ProofError::SecondaryCircuitNotSatisfied))?;

        // Finally, we verify the Spartan proof for the committed relaxed r1cs instance `U_prime`.

        let mut transcript = Transcript::new(b"spartan_snark");
        spartan_snark::SNARK::<Projective<G1>, PC>::verify(
            spartan_proof,
            &key.computation_comm,
            &U_prime.try_into()?,
            &mut transcript,
            &key.snark_gens,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{g1::Config as Bn254Config, Bn254};
    use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, Absorb};
    use ark_ec::short_weierstrass::{Projective, SWCurveConfig};
    use ark_ff::PrimeField;
    use ark_grumpkin::{GrumpkinConfig, Projective as GrumpkinProjective};
    use ark_spartan::polycommitments::{zeromorph::Zeromorph, PolyCommitmentScheme};
    use ark_std::{fs::File, test_rng, One};
    use zstd::stream::Encoder;

    use super::*;
    use crate::{
        circuits::nova::sequential::tests::CubicCircuit,
        commitment::CommitmentScheme,
        nova::sequential::{compression::SNARK, IVCProof, PublicParams},
        pedersen::PedersenCommitment,
        poseidon_config,
    };

    fn test_setup_helper<G1, G2, PC, C2>() -> (
        PC::SRS,
        PublicParams<
            G1,
            G2,
            PVC<G1, PC>,
            C2,
            PoseidonSponge<G1::ScalarField>,
            CubicCircuit<G1::ScalarField>,
        >,
    )
    where
        G1: SWCurveConfig,
        G2: SWCurveConfig<BaseField = G1::ScalarField, ScalarField = G1::BaseField>,
        G1::BaseField: PrimeField + Absorb,
        G2::BaseField: PrimeField + Absorb,
        PC: PolyCommitmentScheme<Projective<G1>>,
        PC::Commitment: Copy + Into<Projective<G1>> + From<Projective<G1>>,
        C2: CommitmentScheme<Projective<G2>, SetupAux = ()>,
    {
        // we hardcode the minimum SRS size here for simplicity. If we need to derive it, we can do:
        // let shape = setup_shape(ro_config, step_circuit).unwrap();
        // let spartan_shape: CRR1CSShape<G1::ScalarField> = shape.clone().try_into().unwrap();
        // let (num_cons, num_vars, num_inputs) = (
        //     spartan_shape.get_num_cons(),
        //     spartan_shape.get_num_vars(),
        //     spartan_shape.get_num_inputs(),
        // );
        // let num_nz_entries = max(shape.A.len(), max(shape.B.len(), shape.C.len()));
        // let min_num_vars = CRSNARKKey::<Projective<G1>, PC>::get_min_num_vars(
        //     num_cons,
        //     num_vars,
        //     num_inputs,
        //     num_nz_entries,
        // )
        // let srs = PC::setup(min_num_vars, b"test_srs", rng).unwrap();
        const NUM_VARS: usize = 26;
        let mut rng = test_rng();
        let ro_config = poseidon_config();
        let step_circuit = CubicCircuit::<G1::ScalarField>::default();
        let srs = PC::setup(NUM_VARS, b"test_srs", &mut rng).unwrap();
        let params = PublicParams::<
            G1,
            G2,
            PVC<G1, PC>,
            C2,
            PoseidonSponge<G1::ScalarField>,
            CubicCircuit<G1::ScalarField>,
        >::setup(ro_config, &step_circuit, &srs, &())
        .expect("setup should not fail");
        (srs, params)
    }

    fn spartan_encode_test_helper<G1, G2, PC, C2>()
    where
        G1: SWCurveConfig,
        G2: SWCurveConfig<BaseField = G1::ScalarField, ScalarField = G1::BaseField>,
        G1::BaseField: PrimeField + Absorb,
        G2::BaseField: PrimeField + Absorb,
        C2: CommitmentScheme<Projective<G2>, SetupAux = ()>,
        PC: PolyCommitmentScheme<Projective<G1>>,
        PC::Commitment: Copy + Into<Projective<G1>> + From<Projective<G1>>,
    {
        // We set up the public parameters both for Nova and Spartan.
        let (srs, params) = test_setup_helper::<G1, G2, PC, C2>();
        let key = SNARK::<
            G1,
            G2,
            PC,
            C2,
            PoseidonSponge<G1::ScalarField>,
            CubicCircuit<G1::ScalarField>,
        >::setup(&params, &srs)
        .unwrap();

        let f = File::create("spartan_key.zst").unwrap();
        let mut enc = Encoder::new(&f, 0).unwrap();
        key.serialize_compressed(&mut enc).unwrap();
        enc.finish().unwrap();
        f.sync_all().unwrap();
    }
    #[test]
    fn spartan_encode_test() {
        spartan_encode_test_helper::<
            Bn254Config,
            GrumpkinConfig,
            Zeromorph<Bn254>,
            PedersenCommitment<GrumpkinProjective>,
        >();
    }
    fn compression_test_helper<G1, G2, PC, C2>()
    where
        G1: SWCurveConfig,
        G2: SWCurveConfig<BaseField = G1::ScalarField, ScalarField = G1::BaseField>,
        G1::BaseField: PrimeField + Absorb,
        G2::BaseField: PrimeField + Absorb,
        C2: CommitmentScheme<Projective<G2>, SetupAux = ()>,
        PC: PolyCommitmentScheme<Projective<G1>>,
        PC::Commitment: Copy + Into<Projective<G1>> + From<Projective<G1>>,
    {
        let circuit = CubicCircuit::<G1::ScalarField>::default();
        // First we set up a Nova PCD instance.
        let z_0 = vec![G1::ScalarField::one()];

        // We set up the public parameters both for Nova and Spartan.
        let (srs, params) = test_setup_helper::<G1, G2, PC, C2>();
        let key = SNARK::<
            G1,
            G2,
            PC,
            C2,
            PoseidonSponge<G1::ScalarField>,
            CubicCircuit<G1::ScalarField>,
        >::setup(&params, &srs)
        .unwrap();

        // Now, we perform a PCD proof step and check that the resulting proof verifies.
        let num_steps = 1;
        let mut nova_proof = IVCProof::new(&z_0);
        nova_proof = nova_proof.prove_step(&params, &circuit).unwrap();
        nova_proof.verify(&params, num_steps).unwrap();

        assert_eq!(&nova_proof.z_i()[0], &G1::ScalarField::from(7));

        // Now, we compress the proof using Spartan
        let compressed_nova_proof = SNARK::<
            G1,
            G2,
            PC,
            C2,
            PoseidonSponge<G1::ScalarField>,
            CubicCircuit<G1::ScalarField>,
        >::compress(&params, &key, nova_proof)
        .unwrap();

        // And check that the compressed proof verifies.
        SNARK::<
                 G1,
                 G2,
                 PC,
                 C2,
                 PoseidonSponge<G1::ScalarField>,
                 CubicCircuit<G1::ScalarField>,
             >::verify(&key, &params, &compressed_nova_proof)
             .unwrap();
    }

    #[test]
    fn compression_test() {
        compression_test_helper::<
            Bn254Config,
            GrumpkinConfig,
            Zeromorph<Bn254>,
            PedersenCommitment<GrumpkinProjective>,
        >();
    }
}
