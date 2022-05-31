import * as wasm from "lurk-rs";
var output = document.getElementById("output");
var status = document.getElementById("status");

function show(text) {
    output.innerText = text;
    console.log(text);
}

function log(text) {
    output.append("\n\n");
    output.append(text);
    console.log(text);
}


function setStatus(iterations) {
  const text = "iterations: " + iterations;
  status.innerText = text;
}

var runit = document.getElementById("run");
var input = document.getElementById("input");
runit.onclick = function (_e) {
    var source = input.value;
    console.log("running: " + source);

    const out = wasm.run_lurk(source);
    const obj = JSON.parse(out);
    setStatus(obj.iterations);
    show(obj.result);

};
