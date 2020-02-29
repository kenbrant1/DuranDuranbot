~front_page_func = {

	~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;


	~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/2.9774535809019));
	~firstviewtext1.background=~backgroundcolor;
	~firstviewtext1.stringColor_(~stringcolor);
	~firstviewtext1.string = "Hello. This is DuranDuranbot. \n\nTo enhance the creativity of this AI bot... please consider providing it with some personal creative inspiration of yours by entering 3 number values into the 3 fields below, numerical values up to 8 digits long, and hit Enter. Consider writing down the 3 values you gave it to be able to replicate exactly the music made by this AI in this specific instance at a later date - although they will still be saved in the "++"duranduranbot_saved_random _seeds.scd".quote++" file, accessible on the next page.";
	~firstviewtext1.align = \topLeft;
	~firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);


	~randseedEZText1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.6, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
	~randseedEZText1.string = "Random Seed 1";
	~randseedEZText1.background_(~backgroundcolor);
	~randseedEZText1.stringColor_(~stringcolor);
	~randseedEZText1.font = Font(size: ~firstviewboundsWIDTH/48);
	~randseedEZText1.action = {
		var randseed_1_make;
		randseed_1_make = ~randseedEZText1.value;
		randseed_1_make = randseed_1_make.asString++"";
		randseed_1_make = randseed_1_make.separate;
		if(randseed_1_make.size > 8, {randseed_1_make = nil; ~randseedEZText1.string = "must be 8 digits or less";}, {
			randseed_1_make = randseed_1_make.join.asString.interpret;
			~randseedviewtext1.string = if(randseed_1_make.isNumber == true, {
				randseed_1_make = randseed_1_make.asFloat; randseed_1_make.postln; ~randseed_1 = randseed_1_make; ~save_random_seeds_func.value; ~randseed_1.asString++" entered";}, {
				randseed_1_make = nil; ~randseedEZText1.string = "Enter Numerical Value";
			});
		});
	};

	~randseedEZText2 = TextField(~firstview, Rect(~firstviewboundsWIDTH/3.7894736842105, ~firstviewboundsHEIGHT/2.6, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
	~randseedEZText2.string = "Random Seed 2";
	~randseedEZText2.background_(~backgroundcolor);
	~randseedEZText2.stringColor_(~stringcolor);
	~randseedEZText2.font = Font(size: ~firstviewboundsWIDTH/48);
	~randseedEZText2.action = {
		var randseed_2_make;
		randseed_2_make = ~randseedEZText2.value;
		randseed_2_make = randseed_2_make.asString++"";
		randseed_2_make = randseed_2_make.separate;
		if(randseed_2_make.size > 8, {randseed_2_make = nil; ~randseedEZText2.string = "must be 8 digits or less";}, {
			randseed_2_make = randseed_2_make.join.asString.interpret;
			~randseedviewtext2.string = if(randseed_2_make.isNumber == true, {
				randseed_2_make = randseed_2_make.asFloat; randseed_2_make.postln; ~randseed_2 = randseed_2_make; ~save_random_seeds_func.value; ~randseed_2.asString++" entered";}, {
				randseed_2_make = nil; ~randseedEZText2.string = "Enter Numerical Value";
			});
		});
	};

	~randseedEZText3 = TextField(~firstview, Rect(~firstviewboundsWIDTH/1.8947368421053, ~firstviewboundsHEIGHT/2.6, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
	~randseedEZText3.string = "Random Seed 3";
	~randseedEZText3.background_(~backgroundcolor);
	~randseedEZText3.stringColor_(~stringcolor);
	~randseedEZText3.font = Font(size: ~firstviewboundsWIDTH/48);
	~randseedEZText3.action = {
		var randseed_3_make;
		randseed_3_make = ~randseedEZText3.value;
		randseed_3_make = randseed_3_make.asString++"";
		randseed_3_make = randseed_3_make.separate;
		if(randseed_3_make.size > 8, {randseed_3_make = nil; ~randseedEZText3.string = "must be 8 digits or less";}, {
			randseed_3_make = randseed_3_make.join.asString.interpret;
			~randseedviewtext3.string = if(randseed_3_make.isNumber == true, {
				randseed_3_make = randseed_3_make.asFloat; randseed_3_make.postln; ~randseed_3 = randseed_3_make; ~save_random_seeds_func.value; ~randseed_3.asString++" entered";}, {
				randseed_3_make = nil; ~randseedEZText3.string = "Enter Numerical Value";
			});
		});
	};

	~randseedviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.26, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
	~randseedviewtext1.background=~backgroundcolor;
	~randseedviewtext1.stringColor_(~stringcolor);
	~randseedviewtext1.string = if(~randseed_1 != nil, {~randseed_1.asString++" entered"}, {"number not entered"});
	~randseedviewtext1.align = \topLeft;
	~randseedviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

	~randseedviewtext2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.7894736842105,~firstviewboundsHEIGHT/2.26, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
	~randseedviewtext2.background=~backgroundcolor;
	~randseedviewtext2.stringColor_(~stringcolor);
	~randseedviewtext2.string = if(~randseed_2 != nil, {~randseed_2.asString++" entered"}, {"number not entered"});
	~randseedviewtext2.align = \topLeft;
	~randseedviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

	~randseedviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.8947368421053, ~firstviewboundsHEIGHT/2.26, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
	~randseedviewtext3.background=~backgroundcolor;
	~randseedviewtext3.stringColor_(~stringcolor);
	~randseedviewtext3.string = if(~randseed_3 != nil, {~randseed_3.asString++" entered"}, {"number not entered"});
	~randseedviewtext3.align = \topLeft;
	~randseedviewtext3.font = Font(size: ~firstviewboundsWIDTH/48);


	~firstviewtext2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.415725806452, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/10.85));
	~firstviewtext2.background=~backgroundcolor;
	~firstviewtext2.stringColor_(~stringcolor);
	~firstviewtext2.string = "Or you can do none of that, and autogenerate all of those things by clicking on "++"Next".quote++" below... Which you have to do regardless :)...";
	~firstviewtext2.align = \topLeft;
	~firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

	~firstviewnextbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/1.2116451016636, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/10.925)).states_([["Next",~stringcolor,~backgroundcolor],["Next",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~randseed_1 == nil, {~randseed_1 =  99999999.rand;});
		if(~randseed_2 == nil, {~randseed_2 =  99999999.rand;});
		if(~randseed_3 == nil, {~randseed_3 =  99999999.rand;});

		if(~startprogram == nil, {~save_random_seeds_func.value; ~startprogram = 1;});

		~main_page_func.value;
	});
	~firstviewnextbutton.font = Font(size: ~firstviewboundsWIDTH/48);

};