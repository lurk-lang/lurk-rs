use ff::PrimeField;
use libipld::cid::Cid;
use serde::{Deserialize, Serialize};

use multihash::Multihash;

const LURK_PREFIX: u64 = 0x10de;

pub trait LurkField: ff::PrimeField {
    // These constants are assumed to be based on some global table like
    // multicodec, ideally extended to include arbitrary precision codecs
    const FIELD_CODEC: u64;
    const HASH_CODEC: u64;
    const LURK_CODEC_PREFIX: u64 = LURK_PREFIX;
    const NUM_BYTES: usize;

    fn from_bytes(bs: &[u8]) -> Option<Self> {
        let mut def: Self::Repr = Self::default().to_repr();
        def.as_mut().copy_from_slice(bs);
        Self::from_repr(def).into()
    }

    // Tags have to be `u32` because we're trying to fit them into `u64`
    // multicodecs without overlapping with the existing table. If we can
    // implement arbitrary-sized codecs though we can relax this from
    // Option<u32>, to a arbitrary sized Vec<u8>.
    fn to_tag(f: Self) -> Option<u32>;

    fn to_multicodec(f: Self) -> Option<u64> {
        let tag: u32 = Self::to_tag(f)?;

        Some(Self::LURK_CODEC_PREFIX << 48 | Self::FIELD_CODEC << 32 | u64::from(tag))
    }

    fn from_multicodec(codec: u64) -> Option<Self> {
        let lurk_prefix = (codec & 0xffff_0000_0000_0000) >> 48;
        let field_prefix = (codec & 0x0000_ffff_0000_0000) >> 32;
        let digest = codec & 0x0000_0000_ffff_ffff;
        if lurk_prefix != Self::LURK_CODEC_PREFIX || field_prefix != Self::FIELD_CODEC {
            None
        } else {
            Some(Self::from(digest as u64))
        }
    }

    fn to_multihash(f: Self) -> Multihash {
        Multihash::wrap(Self::HASH_CODEC, f.to_repr().as_ref()).unwrap()
    }

    fn from_multihash(hash: Multihash) -> Option<Self> {
        Self::from_bytes(hash.digest())
    }

    fn to_cid(tag: Self, digest: Self) -> Option<Cid> {
        let codec = Self::to_multicodec(tag)?;
        Some(Cid::new_v1(codec, Self::to_multihash(digest)))
    }
    fn from_cid(cid: Cid) -> Option<(Self, Self)> {
        let tag = Self::from_multicodec(cid.codec())?;
        let dig = Self::from_multihash(*cid.hash())?;
        Some((tag, dig))
    }
}

impl LurkField for blstrs::Scalar {
    const FIELD_CODEC: u64 = 1;
    const HASH_CODEC: u64 = 2;
    const LURK_CODEC_PREFIX: u64 = LURK_PREFIX;
    const NUM_BYTES: usize = 32;

    fn to_tag(f: Self) -> Option<u32> {
        let bytes: Vec<u8> = f.to_repr().as_ref().to_vec();
        let tag_bytes: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if bytes[4..].iter().all(|x| *x == 0) {
            Some(u32::from_le_bytes(tag_bytes))
        } else {
            None
        }
    }
}

impl LurkField for pasta_curves::Fq {
    const FIELD_CODEC: u64 = 2;
    const HASH_CODEC: u64 = 3;
    const LURK_CODEC_PREFIX: u64 = LURK_PREFIX;
    const NUM_BYTES: usize = 32;

    fn to_tag(f: Self) -> Option<u32> {
        let bytes: Vec<u8> = f.to_repr().as_ref().to_vec();
        let tag_bytes: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if bytes[4..].iter().all(|x| *x == 0) {
            Some(u32::from_le_bytes(tag_bytes))
        } else {
            None
        }
    }
}

impl LurkField for pasta_curves::Fp {
    const FIELD_CODEC: u64 = 3;
    const HASH_CODEC: u64 = 3;
    const LURK_CODEC_PREFIX: u64 = LURK_PREFIX;
    const NUM_BYTES: usize = 32;

    fn to_tag(f: Self) -> Option<u32> {
        let bytes: Vec<u8> = f.to_repr().as_ref().to_vec();
        let tag_bytes: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if bytes[4..].iter().all(|x| *x == 0) {
            Some(u32::from_le_bytes(tag_bytes))
        } else {
            None
        }
    }
}

// For working around the orphan trait impl rule
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FWrap<F: LurkField>(pub F);

impl<F: LurkField> Serialize for FWrap<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes: Vec<u8> = Vec::from(self.0.to_repr().as_ref());
        bytes.serialize(serializer)
    }
}

impl<'de, F: LurkField> Deserialize<'de> for FWrap<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let bytes: Vec<u8> = Vec::deserialize(deserializer)?;
        let f = F::from_bytes(&bytes).ok_or_else(|| {
            D::Error::custom(format!("expected field element as bytes, got {:?}", &bytes))
        })?;
        Ok(FWrap(f))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use blstrs::Scalar as Fr;

    use crate::store::Tag;
    use quickcheck::{Arbitrary, Gen};

    impl<F: LurkField> Arbitrary for FWrap<F> {
        fn arbitrary(_: &mut Gen) -> Self {
            let f = F::random(rand::thread_rng());
            FWrap(f)
        }
    }

    #[quickcheck]
    fn test_bytes_consistency(f1: FWrap<Fr>) -> bool {
        let bytes = f1.0.to_repr().as_ref().to_owned();
        let f2 = <Fr as LurkField>::from_bytes(&bytes);
        Some(f1.0) == f2
    }

    #[quickcheck]
    fn test_tag_consistency(x: Tag) -> bool {
        let f1 = Fr::from(x as u64);
        let tag = <Fr as LurkField>::to_tag(f1).unwrap();
        let f2 = Fr::from(tag as u64);
        f1 == f2 && x as u32 == tag
    }

    #[quickcheck]
    fn test_multicodec_consistency(x: Tag) -> bool {
        let f1 = Fr::from(x as u64);
        let codec = <Fr as LurkField>::to_multicodec(f1).unwrap();
        let f2 = <Fr as LurkField>::from_multicodec(codec);
        println!("x: {:?}", x);
        println!("f1: {}", f1);
        println!("codec: {:0x}", codec);
        println!("f2: {}", f1);
        Some(f1) == f2
    }
    #[quickcheck]
    fn test_multihash_consistency(f1: FWrap<Fr>) -> bool {
        let hash = <Fr as LurkField>::to_multihash(f1.0);
        let f2 = <Fr as LurkField>::from_multihash(hash);
        Some(f1.0) == f2
    }
    #[quickcheck]
    fn test_cid_consistency(args: (Tag, FWrap<Fr>)) -> bool {
        let (tag1, dig1) = args;
        let cid = <Fr as LurkField>::to_cid(Fr::from(tag1 as u64), dig1.0).unwrap();
        if let Some((tag2, dig2)) = <Fr as LurkField>::from_cid(cid) {
            Fr::from(tag1 as u64) == tag2 && dig1.0 == dig2
        } else {
            false
        }
    }
}
