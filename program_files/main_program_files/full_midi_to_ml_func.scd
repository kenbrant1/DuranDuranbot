~full_midi_to_ml_func = {
	if(("instrument_"++(1)++"_full_midi").asSymbol.envirGet != nil, {
		~number_of_instruments.do(s = 0; {
			s=s+1;
			("instrument_"++(s)++"_full_midi_ml").asSymbol.envirPut(
				("instrument_"++(s)++"_full_midi").asSymbol.envirGet;
			);
			("instrument_"++(s)++"_comp_interval_array_ml").asSymbol.envirPut(
				("instrument_"++(s)++"_comp_interval_array").asSymbol.envirGet;
			);
			("instrument_"++(s)++"_comp_choice_array_ml").asSymbol.envirPut(
				("instrument_"++(s)++"_comp_choice_array").asSymbol.envirGet;
			);
		});
	});
};