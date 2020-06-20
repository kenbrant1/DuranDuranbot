~synthdefs_func = {

	~number_of_instruments.do(~synth_inst_num = 0; {

		~synth_inst_num = ~synth_inst_num + 1;
		~synthdefs_func_2.value;

	});

	SynthDef("ddbot_synth", {arg out = 0, bufnum = 0, freq1 = 200,freq2 = 300,freq3 = 400, amp1 = 1,amp2 = 1,amp3 = 1;
		var sig, sig1, sig2, sig3, x;
		sig = Mix(
			case
			{~number_of_instruments == 1}{
				[
					PlayBuf.ar(("synth_num_channels_"++(1)).asSymbol.envirGet, ("buffer_"++(1)).asSymbol.envirGet, freq1/("vd_midi_"++(1)).asSymbol.envirGet.midicps, startPos: 0, loop: 1)*amp1
				]
			}
			{~number_of_instruments == 2}{
				[
					PlayBuf.ar(("synth_num_channels_"++(1)).asSymbol.envirGet, ("buffer_"++(1)).asSymbol.envirGet, freq1/("vd_midi_"++(1)).asSymbol.envirGet.midicps, startPos: 0, loop: 1)*amp1,
					PlayBuf.ar(("synth_num_channels_"++(2)).asSymbol.envirGet, ("buffer_"++(2)).asSymbol.envirGet, freq2/("vd_midi_"++(2)).asSymbol.envirGet.midicps, startPos: 0, loop: 1)*amp2

				]
			}
			{~number_of_instruments == 3}{
				[
					PlayBuf.ar(("synth_num_channels_"++(1)).asSymbol.envirGet, ("buffer_"++(1)).asSymbol.envirGet, freq1/("vd_midi_"++(1)).asSymbol.envirGet.midicps, startPos: 0, loop: 1)*amp1,
					PlayBuf.ar(("synth_num_channels_"++(2)).asSymbol.envirGet, ("buffer_"++(2)).asSymbol.envirGet, freq2/("vd_midi_"++(2)).asSymbol.envirGet.midicps, startPos: 0, loop: 1)*amp2,
					PlayBuf.ar(("synth_num_channels_"++(3)).asSymbol.envirGet, ("buffer_"++(3)).asSymbol.envirGet, freq3/("vd_midi_"++(3)).asSymbol.envirGet.midicps, startPos: 0, loop: 1)*amp3

				]
			};
		);
		/*sig = BBandPass.ar(sig, freq: ((((~midi_high_limit-~midi_low_limit)/2).roundUp + ~midi_low_limit).midicps), bw: /*MouseX.kr(0.01, 8).poll*/0.028, mul: 1, add: 0);
		*//*sig = BPeakEQ.ar(sig, freq: ~vd_midi.midicps, rq: 1, db: 1, mul: 1, add: 0);
		*//*sig = BPF.ar(sig, freq: freq, rq: 1.6, mul: 1, add: 0);*/
		/*sig = BLowPass.ar(sig, freq: 75.midicps, rq: MouseX.kr(0.01, 8).poll /*1*/, mul: 1, add: 0);
		*//*sig = BLowPass4.ar(sig, freq: 75.midicps, rq: MouseX.kr(0.01, 8).poll /*8*/, mul: 1, add: 0);*/
		/*sig = LPZ2.ar(sig, mul: 1, add: 0);*/
		/*sig = BPZ2.ar(sig, mul: 1, add: 0);*/
		/*CompanderD.ar(in: sig, thresh: 0.0001, slopeBelow: 1, slopeAbove: 1, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);*/
		/*Normalizer.ar(in: sig, level: 0.01, dur: 0.01);*/
		/*Limiter.ar(in: sig, level: 0.0001, dur: 0.01);*/
		sig = Compander.ar(in: sig, control: sig,
			thresh: /*MouseY.kr(0.01, 1).poll*/
			case
			{~number_of_instruments == 1}{0.06}
			{~number_of_instruments == 2}{0.06}
			{~number_of_instruments == 3}{0.06};
			,
			slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
		/*sig = Limiter.ar(in: sig, level: 1, dur: 0.01);
		*/
		sig = sig * EnvGen.kr(Env.asr(0.1, 1, 0.1), doneAction: Done.freeSelf);
		OffsetOut.ar(0, sig);
	}).add;
};

~synthdefs_func_2 = {

	case
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Alesis-Sanctuary-QCard-Choir-Aah"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Alesis-Sanctuary-QCard-Choir-Aah-C4-loop.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Alesis-Sanctuary-QCard-Promenade"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Alesis-Sanctuary-QCard-Promenade-C3-loop5a.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(48);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Korg-M1-Angels"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Korg-M1-Angels-C4-loop.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Korg-M3R-Chorale"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Korg-M3R-Chorale-C2_loop.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(36);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Korg-M3R-VoiceChoir"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Korg-M3R-VoiceChoir-C3-loop.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(48);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Korg-Poly-800-Organ"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Korg-Poly-800-Organ-C4_loop.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Korg-Poly-800-samp_2"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Vintage Synth Samples - KORG POLY-800 - 02 KORG POLY-800_loop14.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(1);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Roland-JV-2080-Choir-Aahs"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Roland-JV-2080-Choir-Aahs-C4-loop2.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Roland JX-3P samp 2"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Vintage Synth Samples - ROLAND JX-3P VOL. 1 - 02 ROLAND JX-3P_loop7.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(1);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Yamaha-SY22-Itopia"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Yamaha-SY22-Itopia-C4-loop.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Yamaha-TG500-CH-Aah"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Yamaha-TG500-CH-Aah-C3-1-loop3.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(48);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	}
	{("synth_choice_"++(~synth_inst_num)).asSymbol.envirGet == "Yamaha-TG500-CH-Spirit"}{
		("buffer_"++(~synth_inst_num)).asSymbol.envirPut(
			Buffer.read(Server.default, (~ddbot_root)++"/program_files/other_program_files/samples/Yamaha-TG500-CH-Spirit-C4-loop.wav");

		);
		("vd_midi_"++(~synth_inst_num)).asSymbol.envirPut(60);
		("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirPut(2);
	};

	//~vd_midi - middle note for equalizing volume/amplitude.
	// ~duranduranbotsynth1.isRunning;

	//Synth("synth1").isRunning
	/*~amp = 1;*/
	/*
	{ SinOsc.ar(60.midicps, 0, 0.5) }.play;*/

	/*SynthDef("synth"++(~synth_inst_num), {arg out = 0, bufnum = 0, freq = 1, amp = ("amp_"++(~synth_inst_num)).asSymbol.envirGet;
	var sig;
	sig = PlayBuf.ar(("synth_num_channels_"++(~synth_inst_num)).asSymbol.envirGet, ("buffer_"++(~synth_inst_num)).asSymbol.envirGet, freq/("vd_midi_"++(~synth_inst_num)).asSymbol.envirGet.midicps, startPos: 0, loop: 1);
	/*sig = BBandPass.ar(sig, freq: ((((~midi_high_limit-~midi_low_limit)/2).roundUp + ~midi_low_limit).midicps), bw: /*MouseX.kr(0.01, 8).poll*/0.028, mul: 1, add: 0);
	*//*sig = BPeakEQ.ar(sig, freq: ~vd_midi.midicps, rq: 1, db: 1, mul: 1, add: 0);
	*//*sig = BPF.ar(sig, freq: freq, rq: 1.6, mul: 1, add: 0);*/
	/*sig = BLowPass.ar(sig, freq: 75.midicps, rq: MouseX.kr(0.01, 8).poll /*1*/, mul: 1, add: 0);
	*//*sig = BLowPass4.ar(sig, freq: 75.midicps, rq: MouseX.kr(0.01, 8).poll /*8*/, mul: 1, add: 0);*/
	/*sig = LPZ2.ar(sig, mul: 1, add: 0);*/
	/*sig = BPZ2.ar(sig, mul: 1, add: 0);*/
	/*CompanderD.ar(in: sig, thresh: 0.0001, slopeBelow: 1, slopeAbove: 1, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);*/
	/*Normalizer.ar(in: sig, level: 0.01, dur: 0.01);*/
	/*Limiter.ar(in: sig, level: 0.0001, dur: 0.01);*/
	sig = Compander.ar(in: sig, control: sig,
	thresh: /*MouseY.kr(0.01, 1).poll*/
	case
	{~number_of_instruments == 1}{0.1}
	{~number_of_instruments == 2}{0.1}
	{~number_of_instruments == 3}{0.06};
	,
	slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	sig = sig * EnvGen.kr(Env.asr(0.1, 1, 0.1), doneAction: Done.freeSelf);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*amp);
	}).add;*/

	/*SynthDef("synth4", {arg out = 0, bufnum = 0, freq = 1, amp = 1;
	var sig;
	sig = PlayBuf.ar(1, ~buffer,  BufRateScale.kr(~buffer)*4, startPos: 0, loop: 1);
	/*sig = BBandPass.ar(sig, freq: ((((~midi_high_limit-~midi_low_limit)/2).roundUp + ~midi_low_limit).midicps), bw: /*MouseX.kr(0.01, 8).poll*/0.028, mul: 1, add: 0);
	*//*sig = BPeakEQ.ar(sig, freq: ~vd_midi.midicps, rq: 1, db: 1, mul: 1, add: 0);
	*//*sig = BPF.ar(sig, freq: freq, rq: 1.6, mul: 1, add: 0);*/
	/*sig = BLowPass.ar(sig, freq: 75.midicps, rq: MouseX.kr(0.01, 8).poll /*1*/, mul: 1, add: 0);
	*//*sig = BLowPass4.ar(sig, freq: 75.midicps, rq: MouseX.kr(0.01, 8).poll /*8*/, mul: 1, add: 0);*/
	/*sig = LPZ2.ar(sig, mul: 1, add: 0);*/
	/*sig = BPZ2.ar(sig, mul: 1, add: 0);*/
	/*CompanderD.ar(in: sig, thresh: 0.0001, slopeBelow: 1, slopeAbove: 1, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);*/
	/*Normalizer.ar(in: sig, level: 0.01, dur: 0.01);*/
	/*Limiter.ar(in: sig, level: 0.0001, dur: 0.01);*/
	sig = Compander.ar(in: sig, control: sig, thresh: /*MouseY.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*amp);
	}).play;

	SynthDef("synth4", {arg out = 0, bufnum = 0, freq = 1, amp = 1;
	var sig;
	sig = PlayBuf.ar(1, ~buffer,  BufRateScale.kr(~buffer)*1, startPos: 0, loop: 1);
	sig = Compander.ar(in: sig, control: sig, thresh: /*MouseY.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*amp);
	}).play;
	SynthDef("synth4", {arg out = 0, bufnum = 0, freq = 1, amp = 1;
	var sig;
	sig = PlayBuf.ar(1, ~buffer,  (70.midicps/~vd_midi.midicps), startPos: 0, loop: 1);
	sig = Compander.ar(in: sig, control: sig, thresh: /*MouseY.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*amp);
	}).play;
	x = SynthDef("synth4", {arg out = 0, bufnum = 0, freq = 1, amp = 1;
	var sig;
	sig = PlayBuf.ar(1, ~buffer, BufRateScale.kr(~buffer)*(81.midicps/~vd_midi.midicps), startPos: 0, loop: 1);
	sig = Compander.ar(in: sig, control: sig, thresh: /*MouseY.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*(~amp - (~freq_r.cpsmidi-45*(0.7/36))));
	}).play;
	x.set(\amp, 0.4); 0.3/1 30.midicps/66.midicps 81-45 0.6/36 30-45*(0.6/36)
	SynthDef("synth4", {arg out = 0, bufnum = 0, freq = 1, amp = 1;
	var sig;
	sig = PlayBuf.ar(1, ~buffer,  BufRateScale.kr(~buffer)*(45.midicps/~vd_midi.midicps), startPos: 0, loop: 1);
	sig = Compander.ar(in: sig, control: sig, thresh: /*MouseY.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*amp);
	}).play;
	SynthDef("synth4", {arg out = 0, bufnum = 0, freq = 1, amp = 1;
	var sig;
	sig = PlayBuf.ar(1, ~buffer, (BufRateScale.kr(~buffer)*(40.midicps/~vd_midi.midicps))!2, startPos: 0, loop: 1);
	sig = Compander.ar(in: sig, control: sig, thresh: /*MouseY.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*amp);
	}).play;
	{ SinOsc.ar(60.midicps, 0, 0.2).dup }.play;

	z = Synth("synth1", [freq: 30.midicps, amp: 1]);
	x = Synth("synth1", [freq: 50.midicps]);
	y = Synth("synth2", [freq: 70.midicps, amp: 1]);

	FreqScope.new(400, 200, 0, server: Server.default);*/

	/*x = SynthDef("synth4", {arg out = 0, bufnum = 0, freq = 1, amp = 1;
	var sig;
	sig = PlayBuf.ar(1, ~buffer, BufRateScale.kr(~buffer)*(81.midicps/~vd_midi.midicps), startPos: 0, loop: 1);
	sig = Compander.ar(in: sig, control: sig, thresh: /*MouseY.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.0001, 3.0).poll*/ 1, slopeAbove: /*MouseX.kr(0.0000001, 1).poll*/ 0.31, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
	/*sig = Limiter.ar(in: sig, level: MouseX.kr(0.01, 1).poll, dur: MouseY.kr(0.01, 1).poll);*/
	OffsetOut.ar(0, sig.dup*(1 - (81-45*(0.7/36))));
	}).play;*/

};

~gsine = {

	//starting synthdef setting

	if((~kws_setting == nil) or: (~ratio_setting == nil), {

		~kws_setting = 1; //default type of King Wen sequence interpretation

		~ratio_setting = "psine1"; //default ratio setting for synthdef
	});


	//number of KW(King Wen) sequences in a timewave half-cycle (or amplitude x2)

	~synthdefnum = 1; //default 3


	//timewave starting base (lowest) frequency

	/*x = 432; y = 432; ~basefreqmap = Array.fill(32, {x=x/1.0835058821738}).reverse++[432]++Array.fill(25, {y=y*1.0835058821738}); //create basefreqmap array
	*/
	~base_frequency = ~tuning_frequency/8; //default 8 (63.027950580203). replace "~basefreqmap.at(#)" for custom value.


	//starting synthdef setting

	if((~kws_setting == nil) or: (~ratio_setting == nil), {

		~kws_setting = 1; //default type of King Wen sequence interpretation

		~ratio_setting = "psine1"; //default ratio setting for synthdef
	});


	//volume (amplitude)

	~vol_default = 0.07; //default is 0.0099. For sensitivity reasons, changing this using smaller increments, like 0.001, is recommended. The "if" statement one line below returns any value >= 0.02 to 0.0099.
	/*if(~vol_default >= 0.02, {~vol_default = 0.0099;});*/
	if(~vol == nil, {~vol = ~vol_default});
	12.do(x = 0; {("vol"++(x=x+1)).asSymbol.envirPut(~vol)});
	~number_of_instruments_limit.do(x = 0; {("vol"++(x=x+1)).asSymbol.envirPut(~vol)});


	//low-pass filter settings

	~default_low_pass_filter_freq_limit1 = 1364.0007331375; //default is 1364.0007331375
	~default_low_pass_filter_freq_limit2 = 4841.89; //default is 4841.89

	~default_lpfilter_rq_1 = 1.9277862498129; //default is 1.9277862498129
	~default_lpfilter_rq_2 = 14.9714; //default is 14.9714

	if(~lpfl == nil, {~lpfl = ~default_low_pass_filter_freq_limit1});
	if(~lpf_rq == nil, {~lpf_rq = ~default_lpfilter_rq_1});
	if(~low_pass_filter_on == nil, {~low_pass_filter_on = 1}); //set starting low-pass filter on/off. 1 means on - switch to 0 for off.
	if(~low_pass_filter_freq_limit1 == nil, {~low_pass_filter_freq_limit1 = ~default_low_pass_filter_freq_limit1;});
	if(~low_pass_filter_freq_limit2 == nil, {~low_pass_filter_freq_limit2 = ~default_low_pass_filter_freq_limit2;});
	if(~lpfilter_rq_1 == nil, {~lpfilter_rq_1 = ~default_lpfilter_rq_1;});
	if(~lpfilter_rq_2 == nil, {~lpfilter_rq_2 = ~default_lpfilter_rq_2;});
	~lpfilter_rq_divider1 = 1364.0007331375/1.9277862498129;
	~lpfilter_rq_divider2 = 4841.89/14.9714;
	if(~low_pass_filter_set_value == nil, {~low_pass_filter_freq_limit = ~default_low_pass_filter_freq_limit1;
	});
	if(~low_pass_filter_setting == nil, {~low_pass_filter_setting = 0;});
	if(~rq_set_value == nil, {~rq_set_value = ~default_lpfilter_rq_1});


	//gui set frequency ranges - currently set at the lowest and highest notes on a 128 key range using 432 hz A tuning
	//you can change the tuning near top
	//to insert custom ranges, replace "~freqmap.at()" with a hz frequency

	~outmin1 = ~freqmap.at(0);       // 8.0271480262684 hz                       //base frequency of upper set
	~outmax1 = ~freqmap.at(127);     // 12315.783879572 hz                       //high frequency of upper set
	~outmin2 = ~freqmap.at(0)*8;     // (8.0271480262684)*8 hz                   //base frequency of lower set
	~outmax2 = ~freqmap.at(127)/8;   // (12315.783879572)/8 hz;                  //high frequency of lower set

	~outmina = ~outmin2;
	~outmaxa = ~outmax2;
	~outminb = ~outmin1;
	~outmaxb = ~outmax1;


	//increase or decrease magnitude of second and third harmonics for synthdefs and synth functions (not recommend changing if uncertain of effects)

	//synthdefs

	~first_harmonic = 1;  //1
	~second_harmonic = 2; //2
	~third_harmonic = 6;  //6

	//synth functions

	~icd3 = 3; //3
	~icd6 = 6; //6


	//ratio values

	~fth = 1.4142135623729; //default kws binary interval
	~bi = ~fth; //~bi = "binary interval" - intverval of key change to a hexagram line's opposite in King Wen sequence

	~stepratio = 1.0594630943593;
	~nval = 1.0355417528;
	~n9s = 1.86;
	~n9 = 1.080059789899;
	~n18 = 1.0392592260319;
	~n27 = 1.0260044847071;
	~n36 = 1.0194406437022;
	~n45 = 1.0155225125043;
	~n54 = 1.012918794725;
	~n63 = 1.0110630844869;
	~n72 = 1.0096735332285;

	~gm = 1.6180339887499; //phi (golden mean), scaled on 1 key change per King Wen sequence
	~gm2 = ~gm**(1/2);
	~gm6 = ~gm**(1/6);

	~ngm1 = ~gm**(1/9);    //phi (golden mean), scaled on 9 key changes per King Wen sequence
	~ngm2 = ~ngm1**(1/2);
	~ngm6 = ~ngm1**(1/6);

	~nngm1 = ~gm**(1/64);  //phi (golden mean), scaled on 64 key changes per King Wen sequence
	~nngm2 = ~nngm1**(1/2);
	~nngm6 = ~nngm1**(1/6);

	~h1 = 360/19.47;       //based on 19.47:360, scaled on 1 key change per King Wen sequence
	~h2 = ~h1**(1/2);
	~h6 = ~h1**(1/6);

	~nh1 = ~h1**(1/9);     //based on 19.47:360, scaled on 9 key changes per King Wen sequence
	~nh2 = ~nh1**(1/2);
	~nh6 = ~nh1**(1/6);

	~nnh1 = ~h1**(1/64);   //based on 19.47:360, scaled on 64 key changes per King Wen sequence
	~nnh2 = ~nnh1**(1/2);
	~nnh6 = ~nnh1**(1/6);

	//King Wen sequence interpretation settings 1-3 for synthdefs

	~kws_setting_1_a = {[/*1*/	#a,b,c,d,e,f=[ ~a,~b,~c,~d,~e,~f  ],/*2*/	[ a = a/~bi, b = b/~bi, c = c/~bi, d = d/~bi, e = e/~bi, f = f/~bi ],/*3*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi  ],/*4*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi  ],/*5*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d*~bi,    e = e,         f = f*~bi ],/*6*/	[ a = a*~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f/~bi  ],/*7*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d,         e = e,         f = f ],/*8*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f ],/*9*/	[ a = a*~bi,    b = b,         c = c,         d = d*~bi,    e = e*~bi,    f = f*~bi ],/*10*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f ],/*11*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e,         f = f ],/*12*/	[ a = a*~bi, b = b*~bi, c = c*~bi, d = d/~bi, e = e/~bi, f = f/~bi ],/*13*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e,         f = f*~bi ],/*14*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f],/*15*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d,         e = e/~bi,    f = f/~bi ],/*16*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f ],/*17*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi],/*18*/	[ a = a*~bi, b = b/~bi, c = c/~bi, d = d*~bi, e = e*~bi, f = f/~bi ],/*19*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e,         f = f*~bi ],/*20*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f/~bi ],/*21*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d,         e = e,         f = f*~bi ],/*22*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f ],/*23*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e,         f = f/~bi ],/*24*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi ],/*25*/	[ a = a*~bi,    b = b*~bi,    c = c*~bi,    d = d,         e = e,         f = f ],/*26*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f ],/*27*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f ],/*28*/	[ a = a/~bi, b = b*~bi, c = c*~bi, d = d*~bi, e = e*~bi, f = f/~bi ],/*29*/	[ a = a,         b = b,         c = c/~bi,    d = d/~bi,    e = e,         f = f ],/*30*/	[ a = a*~bi, b = b/~bi, c = c*~bi, d = d*~bi, e = e/~bi, f = f*~bi ],/*31*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f/~bi ],/*32*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f ],/*33*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f ],/*34*/	[ a = a/~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f*~bi ],/*35*/	[ a = a*~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f/~bi ],/*36*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f*~bi ],/*37*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f ],/*38*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f ],/*39*/	[ a = a/~bi, b = b*~bi, c = c/~bi, d = d*~bi, e = e/~bi, f = f/~bi ],/*40*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f ],/*41*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f*~bi ],/*42*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f ],/*43*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d*~bi,    e = e*~bi,    f = f ],/*44*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi ],/*45*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f ],/*46*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f ],/*47*/	[ a = a,         b = b*~bi,    c = c*~bi,    d = d/~bi,    e = e,         f = f ],/*48*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f ],/*49*/	[ a = a,         b = b,         c = c*~bi,    d = d,         e = e/~bi,    f = f*~bi ],/*50*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi ],/*51*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f*~bi ],/*52*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f/~bi ],/*53*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f ],/*54*/	[ a = a/~bi, b = b/~bi, c = c*~bi, d = d/~bi, e = e*~bi, f = f*~bi ],/*55*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e/~bi,    f = f ],/*56*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi ],/*57*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e*~bi,    f = f ],/*58*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f*~bi ],/*59*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f/~bi ],/*60*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi ],/*61*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f ],/*62*/	[ a = a/~bi, b = b/~bi, c = c*~bi, d = d*~bi, e = e/~bi, f = f/~bi ],/*63*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e,         f = f*~bi ],/*64*/	[ a = a*~bi, b = b/~bi, c = c*~bi, d = d/~bi, e = e*~bi, f = f/~bi ]]};
	~kws_setting_1_b = {[/*1*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi ],/*2*/	[ a = a/~bi, b = b/~bi, c = c/~bi, d = d/~bi, e = e/~bi, f = f/~bi ],/*3*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi  ],/*4*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi  ],/*5*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d*~bi,    e = e,         f = f*~bi ],/*6*/	[ a = a*~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f/~bi  ],/*7*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d,         e = e,         f = f ],/*8*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f ],/*9*/	[ a = a*~bi,    b = b,         c = c,         d = d*~bi,    e = e*~bi,    f = f*~bi ],/*10*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f ],/*11*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e,         f = f ],/*12*/	[ a = a*~bi, b = b*~bi, c = c*~bi, d = d/~bi, e = e/~bi, f = f/~bi ],/*13*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e,         f = f*~bi ],/*14*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f],/*15*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d,         e = e/~bi,    f = f/~bi ],/*16*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f ],/*17*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi],/*18*/	[ a = a*~bi, b = b/~bi, c = c/~bi, d = d*~bi, e = e*~bi, f = f/~bi ],/*19*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e,         f = f*~bi ],/*20*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f/~bi ],/*21*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d,         e = e,         f = f*~bi ],/*22*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f ],/*23*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e,         f = f/~bi ],/*24*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi ],/*25*/	[ a = a*~bi,    b = b*~bi,    c = c*~bi,    d = d,         e = e,         f = f ],/*26*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f ],/*27*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f ],/*28*/	[ a = a/~bi, b = b*~bi, c = c*~bi, d = d*~bi, e = e*~bi, f = f/~bi ],/*29*/	[ a = a,         b = b,         c = c/~bi,    d = d/~bi,    e = e,         f = f ],/*30*/	[ a = a*~bi, b = b/~bi, c = c*~bi, d = d*~bi, e = e/~bi, f = f*~bi ],/*31*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f/~bi ],/*32*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f ],/*33*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f ],/*34*/	[ a = a/~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f*~bi ],/*35*/	[ a = a*~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f/~bi ],/*36*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f*~bi ],/*37*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f ],/*38*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f ],/*39*/	[ a = a/~bi, b = b*~bi, c = c/~bi, d = d*~bi, e = e/~bi, f = f/~bi ],/*40*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f ],/*41*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f*~bi ],/*42*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f ],/*43*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d*~bi,    e = e*~bi,    f = f ],/*44*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi ],/*45*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f ],/*46*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f ],/*47*/	[ a = a,         b = b*~bi,    c = c*~bi,    d = d/~bi,    e = e,         f = f ],/*48*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f ],/*49*/	[ a = a,         b = b,         c = c*~bi,    d = d,         e = e/~bi,    f = f*~bi ],/*50*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi ],/*51*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f*~bi ],/*52*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f/~bi ],/*53*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f ],/*54*/	[ a = a/~bi, b = b/~bi, c = c*~bi, d = d/~bi, e = e*~bi, f = f*~bi ],/*55*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e/~bi,    f = f ],/*56*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi ],/*57*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e*~bi,    f = f ],/*58*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f*~bi ],/*59*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f/~bi ],/*60*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi ],/*61*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f ],/*62*/	[ a = a/~bi, b = b/~bi, c = c*~bi, d = d*~bi, e = e/~bi, f = f/~bi ],/*63*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e,         f = f*~bi ],/*64*/	[ a = a*~bi, b = b/~bi, c = c*~bi, d = d/~bi, e = e*~bi, f = f/~bi ]]};

	~kws_setting_2_a = {[/*1*/	#a,b,c,d,e,f=[ ~a,~b,~c,~d,~e,~f  ],/*2*/	[ a = a*~n1/~bi, b = b*~n1/~bi, c = c*~n1/~bi, d = d*~n1/~bi, e = e*~n1/~bi, f = f*~n1/~bi ],/*3*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi    ],/*4*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi    ],/*5*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d*~bi,    e = e,         f = f*~bi    ],/*6*/	[ a = a*~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f/~bi    ],/*7*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d,         e = e,         f = f         ],/*8*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f         ],/*9*/	[ a = a*~bi,    b = b,         c = c,         d = d*~bi,    e = e*~bi,    f = f*~bi    ],/*10*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f         ],/*11*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e,         f = f         ],/*12*/	[ a = a*~n2*~bi, b = b*~n2*~bi, c = c*~n2*~bi, d = d*~n2/~bi, e = e*~n2/~bi, f = f*~n2/~bi ],/*13*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e,         f = f*~bi    ],/*14*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f         ],/*15*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d,         e = e/~bi,    f = f/~bi    ],/*16*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f         ],/*17*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi    ],/*18*/	[ a = a*~n3*~bi, b = b*~n3/~bi, c = c*~n3/~bi, d = d*~n3*~bi, e = e*~n3*~bi, f = f*~n3/~bi ],/*19*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e,         f = f*~bi    ],/*20*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f/~bi    ],/*21*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d,         e = e,         f = f*~bi    ],/*22*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f         ],/*23*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e,         f = f/~bi    ],/*24*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi    ],/*25*/	[ a = a*~bi,    b = b*~bi,    c = c*~bi,    d = d,         e = e,         f = f         ],/*26*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f         ],/*27*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f         ],/*28*/	[ a = a*~n4/~bi, b = b*~n4*~bi, c = c*~n4*~bi, d = d*~n4*~bi, e = e*~n4*~bi, f = f*~n4/~bi ],/*29*/	[ a = a,         b = b,         c = c/~bi,    d = d/~bi,    e = e,         f = f         ],/*30*/	[ a = a*~n5*~bi, b = b*~n5/~bi, c = c*~n5*~bi, d = d*~n5*~bi, e = e*~n5/~bi, f = f*~n5*~bi ],/*31*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f/~bi    ],/*32*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f         ],/*33*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f         ],/*34*/	[ a = a/~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f*~bi    ],/*35*/	[ a = a*~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f/~bi    ],/*36*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f*~bi    ],/*37*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f         ],/*38*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f         ],/*39*/	[ a = a*~n6/~bi, b = b*~n6*~bi, c = c*~n6/~bi, d = d*~n6*~bi, e = e*~n6/~bi, f = f*~n6/~bi ],/*40*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f         ],/*41*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f*~bi    ],/*42*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f         ],/*43*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d*~bi,    e = e*~bi,    f = f         ],/*44*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi    ],/*45*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f         ],/*46*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f         ],/*47*/	[ a = a,         b = b*~bi,    c = c*~bi,    d = d/~bi,    e = e,         f = f         ],/*48*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f         ],/*49*/	[ a = a,         b = b,         c = c*~bi,    d = d,         e = e/~bi,    f = f*~bi    ],/*50*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi    ],/*51*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f*~bi    ],/*52*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f/~bi    ],/*53*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f         ],/*54*/	[ a = a*~n7/~bi, b = b*~n7/~bi, c = c*~n7*~bi, d = d*~n7/~bi, e = e*~n7*~bi, f = f*~n7*~bi ],/*55*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e/~bi,    f = f         ],/*56*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi    ],/*57*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e*~bi,    f = f         ],/*58*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f*~bi    ],/*59*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f/~bi    ],/*60*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi    ],/*61*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f         ],/*62*/	[ a = a*~n8/~bi, b = b*~n8/~bi, c = c*~n8*~bi, d = d*~n8*~bi, e = e*~n8/~bi, f = f*~n8/~bi ],/*63*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e,         f = f*~bi    ],/*64*/	[ a = a*~n9*~bi, b = b*~n9/~bi, c = c*~n9*~bi, d = d*~n9/~bi, e = e*~n9*~bi, f = f*~n9/~bi ]]};
	~kws_setting_2_b = {[/*1*/	[ a = a,         b = b*~bi,    c = c,         d = d*~bi,    e = e,         f = f*~bi    ],/*2*/	[ a = a*~n1/~bi, b = b*~n1/~bi, c = c*~n1/~bi, d = d*~n1/~bi, e = e*~n1/~bi, f = f*~n1/~bi ],/*3*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi    ],/*4*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi    ],/*5*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d*~bi,    e = e,         f = f*~bi    ],/*6*/	[ a = a*~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f/~bi    ],/*7*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d,         e = e,         f = f         ],/*8*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f         ],/*9*/	[ a = a*~bi,    b = b,         c = c,         d = d*~bi,    e = e*~bi,    f = f*~bi    ],/*10*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f         ],/*11*/	[ a = a/~bi,    b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e,         f = f         ],/*12*/	[ a = a*~n2*~bi, b = b*~n2*~bi, c = c*~n2*~bi, d = d*~n2/~bi, e = e*~n2/~bi, f = f*~n2/~bi ],/*13*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e,         f = f*~bi    ],/*14*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f         ],/*15*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d,         e = e/~bi,    f = f/~bi    ],/*16*/	[ a = a,         b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f         ],/*17*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f*~bi    ],/*18*/	[ a = a*~n3*~bi, b = b*~n3/~bi, c = c*~n3/~bi, d = d*~n3*~bi, e = e*~n3*~bi, f = f*~n3/~bi ],/*19*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e,         f = f*~bi    ],/*20*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f/~bi    ],/*21*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d,         e = e,         f = f*~bi    ],/*22*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f         ],/*23*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e,         f = f/~bi    ],/*24*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi    ],/*25*/	[ a = a*~bi,    b = b*~bi,    c = c*~bi,    d = d,         e = e,         f = f         ],/*26*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f         ],/*27*/	[ a = a,         b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f         ],/*28*/	[ a = a*~n4/~bi, b = b*~n4*~bi, c = c*~n4*~bi, d = d*~n4*~bi, e = e*~n4*~bi, f = f*~n4/~bi ],/*29*/	[ a = a,         b = b,         c = c/~bi,    d = d/~bi,    e = e,         f = f         ],/*30*/	[ a = a*~n5*~bi, b = b*~n5/~bi, c = c*~n5*~bi, d = d*~n5*~bi, e = e*~n5/~bi, f = f*~n5*~bi ],/*31*/	[ a = a/~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f/~bi    ],/*32*/	[ a = a,         b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f         ],/*33*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f         ],/*34*/	[ a = a/~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f*~bi    ],/*35*/	[ a = a*~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f/~bi    ],/*36*/	[ a = a/~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f*~bi    ],/*37*/	[ a = a*~bi,    b = b*~bi,    c = c,         d = d,         e = e,         f = f         ],/*38*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f         ],/*39*/	[ a = a*~n6/~bi, b = b*~n6*~bi, c = c*~n6/~bi, d = d*~n6*~bi, e = e*~n6/~bi, f = f*~n6/~bi ],/*40*/	[ a = a,         b = b/~bi,    c = c*~bi,    d = d/~bi,    e = e*~bi,    f = f         ],/*41*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f*~bi    ],/*42*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e/~bi,    f = f         ],/*43*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d*~bi,    e = e*~bi,    f = f         ],/*44*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi    ],/*45*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f         ],/*46*/	[ a = a,         b = b/~bi,    c = c/~bi,    d = d*~bi,    e = e*~bi,    f = f         ],/*47*/	[ a = a,         b = b*~bi,    c = c*~bi,    d = d/~bi,    e = e,         f = f         ],/*48*/	[ a = a,         b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f         ],/*49*/	[ a = a,         b = b,         c = c*~bi,    d = d,         e = e/~bi,    f = f*~bi    ],/*50*/	[ a = a*~bi,    b = b/~bi,    c = c,         d = d,         e = e*~bi,    f = f/~bi    ],/*51*/	[ a = a/~bi,    b = b,         c = c,         d = d/~bi,    e = e/~bi,    f = f*~bi    ],/*52*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d*~bi,    e = e,         f = f/~bi    ],/*53*/	[ a = a,         b = b*~bi,    c = c,         d = d,         e = e,         f = f         ],/*54*/	[ a = a*~n7/~bi, b = b*~n7/~bi, c = c*~n7*~bi, d = d*~n7/~bi, e = e*~n7*~bi, f = f*~n7*~bi ],/*55*/	[ a = a,         b = b,         c = c,         d = d*~bi,    e = e/~bi,    f = f         ],/*56*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f/~bi    ],/*57*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e*~bi,    f = f         ],/*58*/	[ a = a/~bi,    b = b,         c = c*~bi,    d = d/~bi,    e = e,         f = f*~bi    ],/*59*/	[ a = a*~bi,    b = b,         c = c/~bi,    d = d,         e = e,         f = f/~bi    ],/*60*/	[ a = a/~bi,    b = b,         c = c,         d = d,         e = e,         f = f*~bi    ],/*61*/	[ a = a*~bi,    b = b,         c = c,         d = d,         e = e,         f = f         ],/*62*/	[ a = a*~n8/~bi, b = b*~n8/~bi, c = c*~n8*~bi, d = d*~n8*~bi, e = e*~n8/~bi, f = f*~n8/~bi ],/*63*/	[ a = a,         b = b*~bi,    c = c/~bi,    d = d,         e = e,         f = f*~bi    ],/*64*/	[ a = a*~n9*~bi, b = b*~n9/~bi, c = c*~n9*~bi, d = d*~n9/~bi, e = e*~n9*~bi, f = f*~n9/~bi ]]};

	~kws_setting_3_a = {[/*1*/	#a,b,c,d,e,f=[ ~a,~b,~c,~d,~e,~f  ],/*2*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*3*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*4*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1/~bi    ],/*5*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*6*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1/~bi    ],/*7*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*8*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1         ],/*9*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1*~bi    ],/*10*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*11*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1         ],/*12*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*13*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*14*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1         ],/*15*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1,         e = e*~n1/~bi,    f = f*~n1/~bi    ],/*16*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*17*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*18*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1/~bi    ],/*19*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*20*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1/~bi    ],/*21*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*22*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1         ],/*23*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1,         f = f*~n1/~bi    ],/*24*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*25*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*26*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*27*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1         ],/*28*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1/~bi    ],/*29*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1/~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*30*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1*~bi    ],/*31*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*32*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1         ],/*33*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1         ],/*34*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1*~bi    ],/*35*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*36*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*37*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*38*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*39*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*40*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*41*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*42*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1         ],/*43*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*44*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*45*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1         ],/*46*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*47*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*48*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1         ],/*49*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1*~bi,    d = d*~n1,         e = e*~n1/~bi,    f = f*~n1*~bi    ],/*50*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1/~bi    ],/*51*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1*~bi    ],/*52*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1/~bi    ],/*53*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*54*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1*~bi    ],/*55*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1         ],/*56*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*57*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1/~bi,    d = d*~n1,         e = e*~n1*~bi,    f = f*~n1         ],/*58*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*59*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*60*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*61*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*62*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*63*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*64*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1/~bi    ]]};
	~kws_setting_3_b = {[/*1*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*2*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*3*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*4*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1/~bi    ],/*5*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*6*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1/~bi    ],/*7*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*8*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1         ],/*9*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1*~bi    ],/*10*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*11*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1         ],/*12*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*13*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*14*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1         ],/*15*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1,         e = e*~n1/~bi,    f = f*~n1/~bi    ],/*16*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*17*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*18*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1/~bi    ],/*19*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*20*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1/~bi    ],/*21*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*22*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1         ],/*23*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1,         f = f*~n1/~bi    ],/*24*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*25*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*26*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*27*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1         ],/*28*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1/~bi    ],/*29*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1/~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*30*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1*~bi    ],/*31*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*32*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1         ],/*33*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1         ],/*34*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1*~bi    ],/*35*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*36*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*37*/	[ a = a*~n1*~bi,    b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*38*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*39*/	[ a = a*~n1/~bi,    b = b*~n1*~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*40*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*41*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*42*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1/~bi,    f = f*~n1         ],/*43*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*44*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*45*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1         ],/*46*/	[ a = a*~n1,         b = b*~n1/~bi,    c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1*~bi,    f = f*~n1         ],/*47*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1         ],/*48*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1         ],/*49*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1*~bi,    d = d*~n1,         e = e*~n1/~bi,    f = f*~n1*~bi    ],/*50*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1*~bi,    f = f*~n1/~bi    ],/*51*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1/~bi,    e = e*~n1/~bi,    f = f*~n1*~bi    ],/*52*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1*~bi,    e = e*~n1,         f = f*~n1/~bi    ],/*53*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*54*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1*~bi    ],/*55*/	[ a = a*~n1,         b = b*~n1,         c = c*~n1,         d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1         ],/*56*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*57*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1/~bi,    d = d*~n1,         e = e*~n1*~bi,    f = f*~n1         ],/*58*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1,         f = f*~n1*~bi    ],/*59*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1/~bi    ],/*60*/	[ a = a*~n1/~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*61*/	[ a = a*~n1*~bi,    b = b*~n1,         c = c*~n1,         d = d*~n1,         e = e*~n1,         f = f*~n1         ],/*62*/	[ a = a*~n1/~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1*~bi,    e = e*~n1/~bi,    f = f*~n1/~bi    ],/*63*/	[ a = a*~n1,         b = b*~n1*~bi,    c = c*~n1/~bi,    d = d*~n1,         e = e*~n1,         f = f*~n1*~bi    ],/*64*/	[ a = a*~n1*~bi,    b = b*~n1/~bi,    c = c*~n1*~bi,    d = d*~n1/~bi,    e = e*~n1*~bi,    f = f*~n1/~bi    ]]};

	~hsine_ratio = (19.47/360);

	~kws_setting_h_a = {[/*1*/	#a,b,c,d,e,f=[ ~a,~b,~c,~d,~e,~f  ],/*2*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)/~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*3*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi  ],/*4*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)/~bi  ],/*5*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*6*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi  ],/*7*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*8*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*9*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)*~bi ],/*10*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*11*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*12*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)*~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*13*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*14*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)],/*15*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)/~bi ],/*16*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*17*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi],/*18*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)/~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)/~bi ],/*19*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*20*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)/~bi ],/*21*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*22*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*23*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*24*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*25*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*26*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*27*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*28*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)*~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)/~bi ],/*29*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*30*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)*~bi ],/*31*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*32*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*33*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*34*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)*~bi ],/*35*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)/~bi ],/*36*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*37*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*38*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*39*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)*~bi, c = c+(c*~hsine_ratio)/~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*40*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*41*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*42*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*43*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*44*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*45*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*46*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*47*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*48*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*49*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)*~bi ],/*50*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)/~bi ],/*51*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)*~bi ],/*52*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*53*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*54*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)*~bi ],/*55*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*56*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*57*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*58*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*59*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*60*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*61*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*62*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*63*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*64*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)/~bi ]]};
	~kws_setting_h_b = {[/*1*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*2*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)/~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*3*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi  ],/*4*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)/~bi  ],/*5*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*6*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi  ],/*7*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*8*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*9*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)*~bi ],/*10*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*11*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*12*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)*~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*13*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*14*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)],/*15*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)/~bi ],/*16*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*17*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi],/*18*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)/~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)/~bi ],/*19*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*20*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)/~bi ],/*21*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*22*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*23*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*24*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*25*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*26*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*27*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*28*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)*~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)/~bi ],/*29*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*30*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)*~bi ],/*31*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*32*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*33*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*34*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)*~bi ],/*35*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)/~bi ],/*36*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*37*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*38*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*39*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)*~bi, c = c+(c*~hsine_ratio)/~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*40*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*41*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*42*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*43*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*44*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*45*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*46*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*47*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*48*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*49*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)*~bi ],/*50*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio)/~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio)/~bi ],/*51*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio)*~bi ],/*52*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*53*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*54*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)*~bi ],/*55*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio)*~bi,    e = e+(e*~hsine_ratio)/~bi,    f = f+(f*~hsine_ratio) ],/*56*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*57*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio)*~bi,    f = f+(f*~hsine_ratio) ],/*58*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)*~bi,    d = d+(d*~hsine_ratio)/~bi,    e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*59*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)/~bi ],/*60*/	[ a = a+(a*~hsine_ratio)/~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*61*/	[ a = a+(a*~hsine_ratio)*~bi,    b = b+(b*~hsine_ratio),         c = c+(c*~hsine_ratio),         d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio) ],/*62*/	[ a = a+(a*~hsine_ratio)/~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)*~bi, e = e+(e*~hsine_ratio)/~bi, f = f+(f*~hsine_ratio)/~bi ],/*63*/	[ a = a+(a*~hsine_ratio),         b = b+(b*~hsine_ratio)*~bi,    c = c+(c*~hsine_ratio)/~bi,    d = d+(d*~hsine_ratio),         e = e+(e*~hsine_ratio),         f = f+(f*~hsine_ratio)*~bi ],/*64*/	[ a = a+(a*~hsine_ratio)*~bi, b = b+(b*~hsine_ratio)/~bi, c = c+(c*~hsine_ratio)*~bi, d = d+(d*~hsine_ratio)/~bi, e = e+(e*~hsine_ratio)*~bi, f = f+(f*~hsine_ratio)/~bi ]]};


	//set n1-n9 ratio values

	case
	{~ratio_setting == "psine1";}{~n_value1 = ~gm;    ~n_value2 = ~gm2;   ~n_value6 = ~gm6;}
	{~ratio_setting == "psine2";}{~n_value1 = ~ngm1;  ~n_value2 = ~ngm2;  ~n_value6 = ~ngm6;}
	{~ratio_setting == "psine3";}{~n_value1 = ~nngm1; ~n_value2 = ~nngm2; ~n_value6 = ~nngm6;}
	{~ratio_setting == "hsine1";}{~n_value1 = ~gm;    ~n_value2 = ~gm2;   ~n_value6 = ~gm6;};

	//set base (lowest) frequencies, set starting values for synthdef, and load synthdef on server boot

	if((~gsinenum == nil) or: (~synthdef != "gsineicld"), {
		(
			if(~basefreq == nil, {~basefreq = ~base_frequency});
			~fmult = 16;
			(#a,b,c,d,e,f = [ ~basefreq,~basefreq,~basefreq,~basefreq,~basefreq,~basefreq  ];
				~a = a; ~b = b; ~c =c; ~d = d; ~e = e;~f=f;);

			~synthdef = "gsineicld"; ~gsinenum = ~synthdefnum;

			~gsine.value;
		);
	});

	SynthDef(\gsineicfld1, {arg graindur = 0.04, dur = 2000, atk = 0.007, rel = 0.01, gate = 1, amp = 1, i = 1, vol = ~vol, rate = 20, tune = 0,dseqval,
		a, b, c, d, e, f, pan=0, fc = 1, lpfl = ~lpfl, lpf_rq = ~lpf_rq;
		var s1, out, env1, s2, dseq;
		9.do(x = 0; {("n"++(x=x+1)).asSymbol.envirPut(~n_value1)}); //for custom values, replace "~n_value1" with "[1.2, 2.1, etc...].at(x-1)" with the variable amount the same as the #.do value - for all synthdefs
		dseq = Dseq(
			case
			{~kws_setting == 1}{~kws_setting_1_a.value;}
			{~kws_setting == 2}{~kws_setting_2_a.value;}
			{~kws_setting == 3}{~kws_setting_3_a.value;}
			{~kws_setting == 4}{~kws_setting_h_a.value;} ++ Array.fill((~gsinenum*~first_harmonic -1), {
				case
				{~kws_setting == 1}{~kws_setting_1_b.value;}
				{~kws_setting == 2}{~kws_setting_2_b.value;}
				{~kws_setting == 3}{~kws_setting_3_b.value;}
				{~kws_setting == 4}{~kws_setting_h_b.value;}}).flatten, inf);
		s1 = GrainSin.ar(1, Impulse.ar(dur), 1/(dur),Demand.ar(Impulse.ar(dur), 0, dseq));
		if(~low_pass_filter_on == 1, {
			case
			{~low_pass_filter_setting == 0}{s1 = BLowPass.ar(s1, lpfl, lpf_rq);}
			{~low_pass_filter_setting == 1}{s1 = BLowPass4.ar(s1, lpfl, lpf_rq);};
		}, {s1 = s1});
		s1 = Compander.ar(in: s1, control: s1, thresh:/* MouseX.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.01, 1).poll*/ 1, slopeAbove: /*MouseX.kr(0.01, 1).poll*/ 0.28, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
		OffsetOut.ar(0, s1.dup*vol);
	}, [\ir]).add;

	SynthDef(\gsineicrld1, {arg graindur = 0.04, dur = 2000, atk = 0.007, rel = 0.01, gate = 1, amp = 1, i = 1, vol = ~vol, rate = 20, tune = 0,dseqval,
		a, b, c, d, e, f, pan=0, fc = 1, id = -1, lpfl = ~lpfl, lpf_rq = ~lpf_rq;
		var s1, out, env1, s2, dseq;
		9.do(x = 0; {("n"++(x=x+1)).asSymbol.envirPut(~n_value1)});
		dseq = Dseq([
			case
			{~kws_setting == 1}{~kws_setting_1_a.value;}
			{~kws_setting == 2}{~kws_setting_2_a.value;}
			{~kws_setting == 3}{~kws_setting_3_a.value;}
			{~kws_setting == 4}{~kws_setting_h_a.value;} ++ Array.fill((~gsinenum*~first_harmonic -1), {
				case
				{~kws_setting == 1}{~kws_setting_1_b.value;}
				{~kws_setting == 2}{~kws_setting_2_b.value;}
				{~kws_setting == 3}{~kws_setting_3_b.value;}
				{~kws_setting == 4}{~kws_setting_h_b.value;}}).flatten].flatten.reverse, inf);
		s1 = GrainSin.ar(1, Impulse.ar(dur), 1/(dur), Demand.ar(Impulse.ar(dur), 0, dseq));
		if(~low_pass_filter_on == 1, {
			case
			{~low_pass_filter_setting == 0}{s1 = BLowPass.ar(s1, lpfl, lpf_rq);}
			{~low_pass_filter_setting == 1}{s1 = BLowPass4.ar(s1, lpfl, lpf_rq);};
		}, {s1 = s1});
		s1 = Compander.ar(in: s1, control: s1, thresh:/* MouseX.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.01, 1).poll*/ 1, slopeAbove: /*MouseX.kr(0.01, 1).poll*/ 0.28, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
		OffsetOut.ar(0, s1.dup*vol);
	}, [\ir]).add;

	SynthDef(\gsineicfld2, {arg graindur = 0.04, dur = 2000, atk = 0.007, rel = 0.01, gate = 1, amp = 1, i = 1, vol = ~vol, rate = 20, tune = 0,dseqval,
		a, b, c, d, e, f, pan=0, fc = 1, lpfl = ~lpfl, lpf_rq = ~lpf_rq;
		var s1, out, env1, s2, dseq;
		9.do(x = 0; {("n"++(x=x+1)).asSymbol.envirPut(~n_value2)});
		dseq = Dseq(
			case
			{~kws_setting == 1}{~kws_setting_1_a.value;}
			{~kws_setting == 2}{~kws_setting_2_a.value;}
			{~kws_setting == 3}{~kws_setting_3_a.value;}
			{~kws_setting == 4}{~kws_setting_h_a.value;} ++ Array.fill((~gsinenum*~second_harmonic -1), {
				case
				{~kws_setting == 1}{~kws_setting_1_b.value;}
				{~kws_setting == 2}{~kws_setting_2_b.value;}
				{~kws_setting == 3}{~kws_setting_3_b.value;}
				{~kws_setting == 4}{~kws_setting_h_b.value;}}).flatten, inf);
		s1 = GrainSin.ar(1, Impulse.ar(dur), 1/(dur),Demand.ar(Impulse.ar(dur), 0, dseq));
		if(~low_pass_filter_on == 1, {
			case
			{~low_pass_filter_setting == 0}{s1 = BLowPass.ar(s1, lpfl, lpf_rq);}
			{~low_pass_filter_setting == 1}{s1 = BLowPass4.ar(s1, lpfl, lpf_rq);};
		}, {s1 = s1});
		s1 = Compander.ar(in: s1, control: s1, thresh:/* MouseX.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.01, 1).poll*/ 1, slopeAbove: /*MouseX.kr(0.01, 1).poll*/ 0.28, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
		OffsetOut.ar(0, s1.dup*vol);
	}, [\ir]).add;

	SynthDef(\gsineicrld2, {arg graindur = 0.04, dur = 2000, atk = 0.007, rel = 0.01, gate = 1, amp = 1, i = 1, vol = ~vol, rate = 20, tune = 0,dseqval,
		a, b, c, d, e, f, pan=0, fc = 1, id = -1, lpfl = ~lpfl, lpf_rq = ~lpf_rq;
		var s1, out, env1, s2, dseq;
		9.do(x = 0; {("n"++(x=x+1)).asSymbol.envirPut(~n_value2)});
		dseq = Dseq([
			case
			{~kws_setting == 1}{~kws_setting_1_a.value;}
			{~kws_setting == 2}{~kws_setting_2_a.value;}
			{~kws_setting == 3}{~kws_setting_3_a.value;}
			{~kws_setting == 4}{~kws_setting_h_a.value;} ++ Array.fill((~gsinenum*~second_harmonic -1), {
				case
				{~kws_setting == 1}{~kws_setting_1_b.value;}
				{~kws_setting == 2}{~kws_setting_2_b.value;}
				{~kws_setting == 3}{~kws_setting_3_b.value;}
				{~kws_setting == 4}{~kws_setting_h_b.value;}}).flatten].flatten.reverse, inf);
		s1 = GrainSin.ar(1, Impulse.ar(dur), 1/(dur), Demand.ar(Impulse.ar(dur), 0, dseq));
		if(~low_pass_filter_on == 1, {
			case
			{~low_pass_filter_setting == 0}{s1 = BLowPass.ar(s1, lpfl, lpf_rq);}
			{~low_pass_filter_setting == 1}{s1 = BLowPass4.ar(s1, lpfl, lpf_rq);};
		}, {s1 = s1});
		s1 = Compander.ar(in: s1, control: s1, thresh:/* MouseX.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.01, 1).poll*/ 1, slopeAbove: /*MouseX.kr(0.01, 1).poll*/ 0.28, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
		OffsetOut.ar(0, s1.dup*vol);
	}, [\ir]).add;

	SynthDef(\gsineicfld6, {arg graindur = 0.04, dur = 2000, atk = 0.007, rel = 0.01, gate = 1, amp = 1, i = 1, vol = ~vol, rate = 20, tune = 0,dseqval,
		a, b, c, d, e, f, pan=0, fc = 1, lpfl = ~lpfl, lpf_rq = ~lpf_rq;
		var s1, out, env1, s2, dseq;
		9.do(x = 0; {("n"++(x=x+1)).asSymbol.envirPut(~n_value6)});
		dseq = Dseq(
			case
			{~kws_setting == 1}{~kws_setting_1_a.value;}
			{~kws_setting == 2}{~kws_setting_2_a.value;}
			{~kws_setting == 3}{~kws_setting_3_a.value;}
			{~kws_setting == 4}{~kws_setting_h_a.value;} ++ Array.fill((~gsinenum*~third_harmonic -1), {
				case
				{~kws_setting == 1}{~kws_setting_1_b.value;}
				{~kws_setting == 2}{~kws_setting_2_b.value;}
				{~kws_setting == 3}{~kws_setting_3_b.value;}
				{~kws_setting == 4}{~kws_setting_h_b.value;}}).flatten, inf);
		s1 = GrainSin.ar(1, Impulse.ar(dur), 1/(dur),Demand.ar(Impulse.ar(dur), 0, dseq));
		if(~low_pass_filter_on == 1, {
			case
			{~low_pass_filter_setting == 0}{s1 = BLowPass.ar(s1, lpfl, lpf_rq);}
			{~low_pass_filter_setting == 1}{s1 = BLowPass4.ar(s1, lpfl, lpf_rq);};
		}, {s1 = s1});
		s1 = Compander.ar(in: s1, control: s1, thresh:/* MouseX.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.01, 1).poll*/ 1, slopeAbove: /*MouseX.kr(0.01, 1).poll*/ 0.28, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
		OffsetOut.ar(0, s1.dup*vol);
	}, [\ir]).add;

	SynthDef(\gsineicrld6, {arg graindur = 0.04, dur = 2000, atk = 0.007, rel = 0.01, gate = 1, amp = 1, i = 1, vol = ~vol, rate = 20, tune = 0,dseqval,
		a, b, c, d, e, f, pan=0, fc = 1, lpfl = ~lpfl, lpf_rq = ~lpf_rq;
		var s1, out, env1, s2, dseq;
		9.do(x = 0; {("n"++(x=x+1)).asSymbol.envirPut(~n_value6)});
		dseq = Dseq([
			case
			{~kws_setting == 1}{~kws_setting_1_a.value;}
			{~kws_setting == 2}{~kws_setting_2_a.value;}
			{~kws_setting == 3}{~kws_setting_3_a.value;}
			{~kws_setting == 4}{~kws_setting_h_a.value;} ++ Array.fill((~gsinenum*~third_harmonic -1), {
				case
				{~kws_setting == 1}{~kws_setting_1_b.value;}
				{~kws_setting == 2}{~kws_setting_2_b.value;}
				{~kws_setting == 3}{~kws_setting_3_b.value;}
				{~kws_setting == 4}{~kws_setting_h_b.value;}}).flatten].flatten.reverse, inf);
		s1 = GrainSin.ar(1, Impulse.ar(dur), 1/(dur), Demand.ar(Impulse.ar(dur), 0, dseq));
		if(~low_pass_filter_on == 1, {
			case
			{~low_pass_filter_setting == 0}{s1 = BLowPass.ar(s1, lpfl, lpf_rq);}
			{~low_pass_filter_setting == 1}{s1 = BLowPass4.ar(s1, lpfl, lpf_rq);};
		}, {s1 = s1});
		s1 = Compander.ar(in: s1, control: s1, thresh:/* MouseX.kr(0.01, 1).poll*/ 0.1, slopeBelow: /*MouseY.kr(0.01, 1).poll*/ 1, slopeAbove: /*MouseX.kr(0.01, 1).poll*/ 0.28, clampTime: 0.01, relaxTime: 0.1, mul: 1, add: 0);
		OffsetOut.ar(0, s1.dup*vol);
	}, [\ir]).add;

	(//1 - LOAD SYNTH FUNCTIONS
		~synthopen = {
			case
			{((~l1a.isRunning == false) and: (~l1a1.isRunning == false)) or: ~l1a1.isRunning == true}{
				if(~l1a1.isRunning == true, {
					(1.do({
						Server.default.makeBundle(0, {
							(~number_of_instruments).do(x = 0; {
								("l"++(x=x+1)++"a1").asSymbol.envirGet.do(_.set(\dur, 0));
								("l"++(x)++"b1").asSymbol.envirGet.do(_.set(\dur, 0));
								("l"++(x)++"c1").asSymbol.envirGet.do(_.set(\dur, 0));
								("l"++(x)++"d1").asSymbol.envirGet.do(_.set(\dur, 0));
								("l"++(x)++"e1").asSymbol.envirGet.do(_.set(\dur, 0));
								("l"++(x)++"f1").asSymbol.envirGet.do(_.set(\dur, 0));
							});
						});
					}););
				});

				(
					1.do({
						6.do(x = 0; {("s"++(x=x+1)).asSymbol.envirPut([\gsineicfld6, \gsineicrld6, \gsineicfld2, \gsineicrld2, \gsineicfld1, \gsineicrld1].at(x-1))});
						Server.default.makeBundle(0, {
							(~number_of_instruments).do(x = 0; {
								("l"++(x=x+1)++"a").asSymbol.envirPut(Synth(~s1, [\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]).register;);
								("l"++(x)++"b").asSymbol.envirPut(Synth(~s2,     [\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"c").asSymbol.envirPut(Synth(~s3,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"d").asSymbol.envirPut(Synth(~s4,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"e").asSymbol.envirPut(Synth(~s5,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"f").asSymbol.envirPut(Synth(~s6,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
							});
						});
					});
				);

				if(~l1a1.isRunning == true, {
					AppClock.sched(0.161803398875, {
						(x = 0; ((Array.fill(~number_of_instruments, {[
							("l"++(x=x+1)++"a1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"b1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"c1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"d1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"e1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"f1").asSymbol.envirGet.do(_.free);
						]}).flatten).do(_.free))); ~slideroutine.stop;
					});
				});
			}
			{~l1a.isRunning == true}{
				(1.do({
					Server.default.makeBundle(0, {
						(~number_of_instruments).do(x = 0; {
							("l"++(x=x+1)++"a").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"b").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"c").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"d").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"e").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"f").asSymbol.envirGet.do(_.set(\dur, 0));
						});
					});
				}););

				(
					1.do({
						6.do(x = 0; {("s"++(x=x+1)).asSymbol.envirPut([\gsineicfld6, \gsineicrld6, \gsineicfld2, \gsineicrld2, \gsineicfld1, \gsineicrld1].at(x-1))});
						Server.default.makeBundle(0, {
							(~number_of_instruments).do(x = 0; {
								("l"++(x=x+1)++"a1").asSymbol.envirPut(Synth(~s1, [\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]).register;);
								("l"++(x)++"b1").asSymbol.envirPut(Synth(~s2,     [\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"c1").asSymbol.envirPut(Synth(~s3,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"d1").asSymbol.envirPut(Synth(~s4,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"e1").asSymbol.envirPut(Synth(~s5,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
								("l"++(x)++"f1").asSymbol.envirPut(Synth(~s6,     [\dur, ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet]););
							});
						});
					});
				);
				AppClock.sched(0.161803398875, {
					(x = 0; ((Array.fill(~number_of_instruments, {[
						("l"++(x=x+1)++"a").asSymbol.envirGet.do(_.free);
						("l"++(x)++"b").asSymbol.envirGet.do(_.free);
						("l"++(x)++"c").asSymbol.envirGet.do(_.free);
						("l"++(x)++"d").asSymbol.envirGet.do(_.free);
						("l"++(x)++"e").asSymbol.envirGet.do(_.free);
						("l"++(x)++"f").asSymbol.envirGet.do(_.free);
					]}).flatten).do(_.free))); ~slideroutine.stop;
				});
			};
		};

		~synthflow = {
			case
			{~l1a1.isRunning == true}{
				(1.do({
					Server.default.makeBundle(0, {
						(~number_of_instruments).do(x = 0; {
							("l"++(x=x+1)++"a1").asSymbol.envirGet.do(_.set(\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"b1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"c1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"d1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"e1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"f1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
						});
					});
				}););
			}
			{~l1a.isRunning == true} {
				(1.do({
					Server.default.makeBundle(0, {
						(~number_of_instruments).do(x = 0; {
							("l"++(x=x+1)++"a").asSymbol.envirGet.do(_.set(\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"b").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"c").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"d").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"e").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
							("l"++(x)++"f").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
						});
					});
				}););
			};
		};

		~synthslide = {
			case
			{~l1a1.isRunning == true}{(
				~slideroutine = Routine({1.do({
					~slidedo.do({~slidecount=~slidecount-1;
						(
							~number_of_instruments.do(x = 0; {("f"++(x=x+1)).asSymbol.envirPut(("f"++(x)).asSymbol.envirGet + ("sf"++(x)).asSymbol.envirGet)});
							Server.default.makeBundle(0, {
								(~number_of_instruments).do(x = 0; {
									("l"++(x=x+1)++"a1").asSymbol.envirGet.do(_.set(\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"b1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"c1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"d1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"e1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"f1").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
								});
							});
						);

						(~slidetime/~slidedo).wait;});
			});}).play;);}
			{~l1a.isRunning == true} {(
				~slideroutine = Routine({1.do({
					~slidedo.do({~slidecount=~slidecount-1;
						(
							~number_of_instruments.do(x = 0; {("f"++(x=x+1)).asSymbol.envirPut(("f"++(x)).asSymbol.envirGet + ("sf"++(x)).asSymbol.envirGet)});
							Server.default.makeBundle(0, {
								(~number_of_instruments).do(x = 0; {
									("l"++(x=x+1)++"a").asSymbol.envirGet.do(_.set(\dur, ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"b").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet,       \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"c").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"d").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd3, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"e").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
									("l"++(x)++"f").asSymbol.envirGet.do(_.set(\dur,     ("f"++(x)).asSymbol.envirGet/~icd6, \lpfl, ~lpfl, \lpf_rq, ~lpf_rq, \vol, ("vol"++(x)).asSymbol.envirGet));
								});
							});

						);

						(~slidetime/~slidedo).wait;});
			});}).play;);};
		};


		~synthpause = {(
			case
			{~l1a1.isRunning == true}{
				(1.do({
					Server.default.makeBundle(0, {
						(~number_of_instruments).do(x = 0; {
							("l"++(x=x+1)++"a1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"b1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"c1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"d1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"e1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"f1").asSymbol.envirGet.do(_.set(\dur, 0));
						});
					});
				}););
			}
			{~l1a.isRunning == true} {
				(1.do({
					Server.default.makeBundle(0, {
						(~number_of_instruments).do(x = 0; {
							("l"++(x=x+1)++"a").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"b").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"c").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"d").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"e").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"f").asSymbol.envirGet.do(_.set(\dur, 0));
						});
					});
				}););
			};
		);
		};

		~synthfree = {
			(
				case
				{~l1a1.isRunning == true}{
					(1.do({
						(~number_of_instruments).do(x = 0; {
							("l"++(x=x+1)++"a1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"b1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"c1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"d1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"e1").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"f1").asSymbol.envirGet.do(_.set(\dur, 0));
						});
					}););
					AppClock.sched(0.161803398875, {
						(x = 0; ((Array.fill(~number_of_instruments, {[
							("l"++(x=x+1)++"a1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"b1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"c1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"d1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"e1").asSymbol.envirGet.do(_.free);
							("l"++(x)++"f1").asSymbol.envirGet.do(_.free);
						]}).flatten).do(_.free))); ~slideroutine.stop;
					});
				}
				{~l1a.isRunning == true}{
					(1.do({
						(~number_of_instruments).do(x = 0; {
							("l"++(x=x+1)++"a").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"b").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"c").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"d").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"e").asSymbol.envirGet.do(_.set(\dur, 0));
							("l"++(x)++"f").asSymbol.envirGet.do(_.set(\dur, 0));
						});
					}););
					AppClock.sched(0.161803398875, {
						(x = 0; ((Array.fill(~number_of_instruments, {[
							("l"++(x=x+1)++"a").asSymbol.envirGet.do(_.free);
							("l"++(x)++"b").asSymbol.envirGet.do(_.free);
							("l"++(x)++"c").asSymbol.envirGet.do(_.free);
							("l"++(x)++"d").asSymbol.envirGet.do(_.free);
							("l"++(x)++"e").asSymbol.envirGet.do(_.free);
							("l"++(x)++"f").asSymbol.envirGet.do(_.free);
						]}).flatten).do(_.free))); ~slideroutine.stop;
					});
				};
			);
		};
	);

	if((~l1a == nil) or: (~l1a1 == nil), {
		~l1a = Synth(\gsineicfld, [\dur, 0]).register;~l1a1 = Synth(\gsineicfld, [\dur, 0]).register;[~l1a, ~l1a1].do(_.free);~l1a = Synth(\gsineicfld, [\dur, 0]).register;~l1a1 = Synth(\gsineicfld, [\dur, 0]).register;[~l1a, ~l1a1].do(_.free);
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