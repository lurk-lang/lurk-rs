use libipld::{
    cbor::DagCborCodec,
    cid::{
        multihash::{Code, MultihashDigest},
        Cid,
    },
    codec::Codec,
    Ipld,
};
use reqwest::multipart;
use serde_json;

// Store IPLD data on IPFS and get its CID
pub async fn dag_put(host: &String, dag: Ipld) -> Result<String, reqwest::Error> {
    let url = format!(
        "http://{}{}?{}",
        host, "/api/v0/dag/put", "input-codec=dag-cbor&pin=true&hash=blake2b-256"
    );
    let cbor = DagCborCodec.encode(&dag).unwrap();
    let client = reqwest::Client::new();
    let form = multipart::Form::new().part("file", multipart::Part::bytes(cbor.clone()));
    let response: serde_json::Value = client
        .post(url)
        .multipart(form)
        .send()
        .await?
        .json()
        .await?;

    let ipfs_cid: String = response["Cid"]["/"].as_str().unwrap().to_string();
    let local_cid: String = Cid::new_v1(0x71, Code::Blake2b256.digest(&cbor)).to_string();

    if ipfs_cid == local_cid {
        Ok(ipfs_cid)
    } else {
        panic!("CIDs are different {} != {}", ipfs_cid, local_cid);
    }
}

// Load IPLD data from IPFS given a CID
pub async fn dag_get(host: &String, cid: &String) -> Result<Ipld, reqwest::Error> {
    let url = format!("http:/{}{}?arg={}", host, "/api/v0/block/get", cid);
    let client = reqwest::Client::new();
    let response = client.post(url).send().await?.bytes().await?;
    let ipld = DagCborCodec.decode(&response).expect("Invalid CBOR data");

    Ok(ipld)
}

#[cfg(test)]
mod test {
    use crate::ipfs::{dag_get, dag_put};
    use blstrs::Scalar as Fr;
    use libipld::serde::{from_ipld, to_ipld};
    use lurk::scalar_store::ScalarStore;
    use lurk::store::Store;
    use reqwest;

    #[ignore]
    #[tokio::test]
    async fn ipfs_roundtrip() -> Result<(), reqwest::Error> {
        let mut store_in = Store::<Fr>::default();
        let expr = store_in.read("symbol").unwrap();
        store_in.hydrate_scalar_cache();
        let (scalar_store_in, _) = ScalarStore::new_with_expr(&store_in, &expr);

        let ipld = to_ipld(scalar_store_in.clone()).unwrap();
        let cid = dag_put(&String::from("localhost:5001"), ipld).await?;

        let ipld_out = dag_get(&String::from("localhost:5001"), &cid).await?;
        let scalar_store_out = from_ipld(ipld_out).unwrap();

        assert_eq!(scalar_store_in, scalar_store_out);
        Ok(())
    }
}
