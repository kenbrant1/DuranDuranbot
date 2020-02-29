~synthdefs_func = {

	if(~buffer == nil, {~buffer = Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Alesis-Sanctuary-QCard-Promenade-C3-loop5.wav");});

	~vd_midi = 73; //middle note for equalizing volume/amplitude. default is 73.

	/*~amp = 1;*/

	~number_of_instruments.do(x = 0; {
		SynthDef("synth"++(x=x+1), {arg out = 0, bufnum = 0, freq = 1, amp = 1;
			var sig;
			sig = PlayBuf.ar(1, ~buffer, freq/~vd_midi.midicps, startPos: 0, loop: 1);
			sig = sig*amp;
			/*sig = BBandPass.ar(sig, freq: ((((~midi_high_limit-~midi_low_limit)/2).roundUp + ~midi_low_limit).midicps), bw: 1, mul: 1, add: 0);
			*//*sig = BPeakEQ.ar(sig, freq: ~vd_midi.midicps, rq: 1, db: 1, mul: 1, add: 0);
			*//*sig = BPF.ar(sig, freq: freq, rq: 1.6, mul: 1, add: 0);*/
			/*sig = BLowPass.ar(sig, freq: ~vd_midi.midicps, rq: 1, mul: 1, add: 0);*/
			/*sig = BLowPass4.ar(sig, freq: ~vd_midi.midicps, rq: 8, mul: 1, add: 0);*/
			/*sig = LPZ2.ar(sig, mul: 1, add: 0);*/
			/*sig = BPZ2.ar(sig, mul: 1, add: 0);*/
			/*CompanderD.ar(in: sig, thresh: 0.0001, slopeBelow: 1, slopeAbove: 1, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);*/
			/*Normalizer.ar(in: sig, level: 0.01, dur: 0.01);*/
			/*Limiter.ar(in: sig, level: 0.0001, dur: 0.01);*/
			sig = Compander.ar(in: sig, control: sig, thresh:/* MouseX.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.01, 1).poll*/ 1, slopeAbove: /*MouseX.kr(0.01, 1).poll*/ 0.28, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
			Out.ar(0, sig.dup);
		}).add;
	});

};

// old synthdefs

/*	~klangarray1 = {Array.rand(3, 0, 20);};

	~number_of_instruments.do(x = 0; {
	SynthDef("synth"++(x=x+1), {arg freq = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
	var sig;
	sig = Mix(DynKlang.ar(`[ [freq]++~klangarray1.value+freq, nil, nil ], 1, 0) * 0.05);
	sig = RLPF.ar(sig, ffreq, rq);
	sig = sig * EnvGen.kr(
	Env.adsr(0.01, 0.1, 0.6, 0.1),
	gt
	);

	Out.ar(0, sig!2);
	}).play;
	});*/


	/*{ SinOsc.ar(73.midicps, 0, 0.5) }.play;*/

	/*SynthDef("synth1", {| out = 0, bufnum = 0, freq = 1 |
	var sig;
	sig = PlayBuf.ar(1, ~buffer, freq/73.midicps, startPos: 0, loop: 1);
	Out.ar(1, sig!2);
	}).play;*/



	/*~piecedur_secs.postln;
	~number_of_instruments.do(x = 0; {
	SynthDef("synth"++(x=x+1), {
	arg freq = 500, mRatio = 1, cRatio = 1, index = 1, iScale = 5, amp = 0.2, atk = 0.01, rel = 300, cAtk = 4, cRel = (-4), pan = 0;
	var car, mod, env, iEnv;
	iEnv = EnvGen.kr(
	Env.new(
	[index, index * iScale, index],
	[atk, rel],
	[cAtk, cRel]
	)
	);
	env = EnvGen.kr(
	Env.perc(atk, rel, curve: [cAtk, cRel])/*,
	doneAction: 2*/
	);
	mod = SinOsc.ar(freq * mRatio, mul: freq * mRatio * iEnv);
	car = SinOsc.ar(freq * cRatio + mod) * env * amp;
	car = Pan2.ar(car, pan);
	Out.ar(0, car);
	}).add;
	});

SynthDef(\bass_synth1, {arg freq = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
			var sig;
			sig = Mix(DynKlang.ar(`[ [freq]++~klangarray1+freq, nil, nil ], 1, 0) * 0.05);
			sig = RLPF.ar(sig, ffreq, rq);
			sig = sig * EnvGen.kr(
			Env.adsr(0.01, 0.1, 0.6, 0.1),
			gt
			);

			Out.ar(0, sig!2);
			}).add;
			SynthDef(\keys_synth2, {arg freqa = 432, freqb = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
			var sig;
			sig = Mix([DynKlang.ar(`[ [freqa]++~klangarray2+freqa, nil, nil ], 1, 0), DynKlang.ar(`[ [freqb]++~klangarray2+freqb, nil, nil ], 1, 0)]  * 0.05);
			sig = RLPF.ar(sig, ffreq, rq);
			sig = sig * EnvGen.kr(
			Env.adsr(0.01, 0.1, 0.6, 0.1),
			gt
			);

			Out.ar(0, sig!2);
			}).add;

			/*Synth(\keys_synth2, [\freqa, 432, \freqb, 300])*/

			SynthDef(\vocal_synth3, {arg freq = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
			var sig;
			sig = Mix(DynKlang.ar(`[ [freq]++~klangarray3+freq, nil, nil ], 1, 0) * 0.05);
			sig = RLPF.ar(sig, ffreq, rq);
			sig = sig * EnvGen.kr(
			Env.adsr(0.01, 0.1, 0.6, 0.1),
			gt
			);

			Out.ar(0, sig!2);
			}).add;*/