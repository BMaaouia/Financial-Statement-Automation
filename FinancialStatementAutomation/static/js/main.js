//Global var
var CRUMINA = {};

(function ($) {

	// USE STRICT
	"use strict";

	//----------------------------------------------------/
	// Predefined Variables
	//----------------------------------------------------/
	var $window = $(window),
		$document = $(document),
		$body = $('body'),

		swipers = {},
		//Elements
		$header = $('#site-header'),
		$counter = $('.counter'),
		$progress_bar = $('.skills-item'),
		$pie_chart = $('.pie-chart'),
		$animatedIcons = $('.js-animate-icon'),
		$asidePanel = $('.right-menu'),
		$primaryMenu = $('.primary-menu'),
		$footer = $('#site-footer'),
		$preloader = $('#hellopreloader');


	/* -----------------------
     * Header Spacer
     * --------------------- */

	CRUMINA.headerSpacer = {
		$spacer: null,
		$header: null,
		init: function () {
			this.$header = jQuery($header);

			this.$spacer = jQuery('<div class="header--spacer"></div>').insertAfter(this.$header);
		},
		resize: function () {
			var _this = this;

			if (!_this.$spacer) {
				return;
			}

			setTimeout(function () {
				var height = _this.$header.outerHeight();
				var backgroundHeader = _this.$header.css('background-color');
				_this.$spacer.css({'height': height + 'px', 'background-color': backgroundHeader});
			}, 100);
		}
	};


	CRUMINA.updateResponsiveInit = function () {
		var resizeTimer = null;
		var resize = function () {
			resizeTimer = null;

			// Methods
			CRUMINA.headerSpacer.resize();
		};

		$(window).on('resize', function () {
			if (resizeTimer === null) {
				resizeTimer = window.setTimeout(function () {
					resize();
				}, 200);
			}
		}).resize();
	};


	/* -----------------------
 * Preloader
 * --------------------- */

	CRUMINA.preloader = function () {
		$window.scrollTop(0);
		setTimeout(function () {
			$preloader.fadeOut(800);
		}, 800);
		return false;
	};


	var $popupSearch = jQuery(".popup-search");
	var $cartPopap = jQuery(".cart-popup-wrap");


	/* -----------------------
	 * Fixed Header
	 * --------------------- */

	CRUMINA.fixedHeader = function () {
		// grab an element
		$header.headroom(
			{
				"offset": 20,
				"tolerance": 5,
				"classes": {
					"initial": "animated",
					"pinned": "swingInX",
					"unpinned": "swingOutX"
				}
			}
		);
	};

	/* -----------------------
	 * Parallax footer
	 * --------------------- */

	CRUMINA.parallaxFooter = function () {
		if ($footer.length && $footer.hasClass('js-fixed-footer')) {
			$footer.before('<div class="block-footer-height"></div>');
			$('.block-footer-height').matchHeight({
				target: $footer
			});
		}
	};

	/* -----------------------
	 * COUNTER NUMBERS
	 * --------------------- */
	CRUMINA.counters = function () {
		if ($counter.length) {
			$counter.each(function () {
				jQuery(this).waypoint(function () {
					$(this.element).find('span').countTo();
					this.destroy();
				}, {offset: '95%'});
			});
		}
	};

	/* -----------------------
	 * Progress bars Animation
	 * --------------------- */
	CRUMINA.progresBars = function () {
		if ($progress_bar.length) {
			$progress_bar.each(function () {
				jQuery(this).waypoint(function () {
					$(this.element).find('.count-animate').countTo();
					$(this.element).find('.skills-item-meter-active').fadeTo(300, 1).addClass('skills-animate');
					this.destroy();
				}, {offset: '90%'});
			});
		}
	};

	/* -----------------------
	 * Pie chart Animation
	 * --------------------- */
	CRUMINA.pieCharts = function () {
		if ($pie_chart.length) {
			$pie_chart.each(function () {
				jQuery(this).waypoint(function () {
					var current_cart = $(this.element);
					var startColor = current_cart.data('start-color');
					var endColor = current_cart.data('end-color');
					var counter = current_cart.data('value') * 100;

					current_cart.circleProgress({
						thickness: 16,
						size: 320,
						startAngle: -Math.PI / 4 * 2,
						emptyFill: '#fff',
						lineCap: 'round',
						fill: {
							gradient: [startColor, endColor],
							gradientAngle: Math.PI / 4
						}
					}).on('circle-animation-progress', function (event, progress) {
						current_cart.find('.content').html(parseInt(counter * progress, 10) + '<span>%</span>'
						)
					}).on('circle-animation-end', function () {

					});
					this.destroy();

				}, {offset: '90%'});
			});
		}
	};
	/* -----------------------
	 * Animate SVG Icons
	 * --------------------- */
	CRUMINA.animateSvg = function () {
		if ($animatedIcons.length) {
			$animatedIcons.each(function () {
				jQuery(this).waypoint(function () {
					var mySVG = $(this.element).find('> svg').drawsvg();
					mySVG.drawsvg('animate');
					this.destroy();
				}, {offset: '95%'});
			});
		}
	};
	/* -----------------------------
	 * Custom Scroll bar
	 * ---------------------------*/
	CRUMINA.customScroll = function () {
		if ($asidePanel.length) {
			$asidePanel.mCustomScrollbar({
				axis: "y",
				scrollEasing: "linear",
				scrollInertia: 150
			});
		}
	};
	/* -----------------------------
	 * Toggle aside panel on click
	 * ---------------------------*/
	CRUMINA.togglePanel = function () {
		if ($asidePanel.length) {
			$asidePanel.toggleClass('opened');
			$body.toggleClass('overlay-enable');
		}
	};
	/* -----------------------------
	 * Toggle search overlay
	 * ---------------------------*/
	CRUMINA.toggleSearch = function () {
		$body.toggleClass('open');
		$('.overlay_search-input').focus();
	};
	/* -----------------------------
	 * Embedded Video in pop up
	 * ---------------------------*/
	CRUMINA.mediaPopups = function () {
		$('.js-popup-iframe').magnificPopup({
			disableOn: 700,
			type: 'iframe',
			mainClass: 'mfp-fade',
			removalDelay: 160,
			preloader: false,

			fixedContentPos: false,

			iframe: {
				patterns: {
					youtube: {
						src: 'https://www.youtube.com/embed/%id%?autoplay=1'
					},
					vimeo: {
						src: 'https://player.vimeo.com/video/%id%?autoplay=1'
					},
				}
			}
		});
		$('.js-zoom-image, .link-image').magnificPopup({
			type: 'image',
			removalDelay: 500, //delay removal by X to allow out-animation
			callbacks: {
				beforeOpen: function () {
					// just a hack that adds mfp-anim class to markup
					this.st.image.markup = this.st.image.markup.replace('mfp-figure', 'mfp-figure mfp-with-anim');
					this.st.mainClass = 'mfp-zoom-in';
				}
			},
			closeOnContentClick: true,
			midClick: true
		});
	};

	/* -----------------------------
	 * Equal height
	 * ---------------------------*/
	CRUMINA.equalHeight = function () {
		$('.js-equal-child').find('.theme-module').matchHeight({
			property: 'min-height'
		});
	};

	/* -----------------------------
	 * Scrollmagic scenes animation
	 * ---------------------------*/
	CRUMINA.SubscribeScrollAnnimation = function () {
		var controller = new ScrollMagic.Controller();
		new ScrollMagic.Scene({triggerElement: ".subscribe"})
			.setVelocity(".gear", {opacity: 1, rotateZ: "360deg"}, 1200)
			.triggerHook("onEnter")
			.addTo(controller);

		new ScrollMagic.Scene({triggerElement: ".subscribe"})
			.setVelocity(".mail", {opacity: 1, bottom: "0"}, 600)
			.triggerHook(0.8)
			.addTo(controller);

		new ScrollMagic.Scene({triggerElement: ".subscribe"})
			.setVelocity(".mail-2", {opacity: 1, right: "20"}, 800)
			.triggerHook(0.9)
			.addTo(controller);
	};

	CRUMINA.SeoScoreScrollAnnimation = function () {
		var controller = new ScrollMagic.Controller();

		new ScrollMagic.Scene({triggerElement: ".seo-score"})
			.setVelocity(".seo-score .images img:first-of-type", {opacity: 1, top: "-10"}, 400)
			.triggerHook("onEnter")
			.addTo(controller);

		new ScrollMagic.Scene({triggerElement: ".seo-score"})
			.setVelocity(".seo-score .images img:nth-child(2)", {opacity: 1, bottom: "0"}, 800)
			.triggerHook(0.7)
			.addTo(controller);

		new ScrollMagic.Scene({triggerElement: ".seo-score"})
			.setVelocity(".seo-score .images img:last-child", {opacity: 1, bottom: "0"}, 1000)
			.triggerHook(0.8)
			.addTo(controller);
	};

	CRUMINA.TestimonialScrollAnnimation = function () {
		var controller = new ScrollMagic.Controller();

		new ScrollMagic.Scene({triggerElement: ".testimonial-slider"})
			.setVelocity(".testimonial-slider .testimonial-img", {opacity: 1, bottom: "-50"}, 400)
			.triggerHook(0.6)
			.addTo(controller);

		new ScrollMagic.Scene({triggerElement: ".testimonial-slider"})
			.setVelocity(".testimonial-slider .testimonial__thumb-img", {opacity: 1, top: "-100"}, 600)
			.triggerHook(1)
			.addTo(controller);
	};

	CRUMINA.OurVisionScrollAnnimation = function () {
		var controller = new ScrollMagic.Controller();

		new ScrollMagic.Scene({triggerElement: ".our-vision"})
			.setVelocity(".our-vision .elements", {opacity: 1}, 600)
			.triggerHook(0.6)
			.addTo(controller);

		new ScrollMagic.Scene({triggerElement: ".our-vision"})
			.setVelocity(".our-vision .eye", {opacity: 1, bottom: "-90"}, 1000)
			.triggerHook(1)
			.addTo(controller);
	};

	CRUMINA.MountainsScrollAnnimation = function () {
		var controller = new ScrollMagic.Controller();

		new ScrollMagic.Scene({triggerElement: ".background-mountains"})
			.setVelocity(".images img:first-child", {opacity: 1, bottom: "0", paddingBottom: "10%"}, 800)
			.triggerHook(0.4)
			.addTo(controller);

		new ScrollMagic.Scene({triggerElement: ".background-mountains"})
			.setVelocity(".images img:last-child", {opacity: 1, bottom: "0"}, 800)
			.triggerHook(0.3)
			.addTo(controller);
	};
	/* -----------------------------
	 * Isotope sorting
	 * ---------------------------*/

	CRUMINA.IsotopeSort = function () {
		var $container = $('.sorting-container');
		$container.each(function () {
			var $current = $(this);
			var layout = ($current.data('layout').length) ? $current.data('layout') : 'masonry';
			$current.isotope({
				itemSelector: '.sorting-item',
				layoutMode: layout,
				percentPosition: true
			});

			$current.imagesLoaded().progress(function () {
				$current.isotope('layout');
			});

			var $sorting_buttons = $current.siblings('.sorting-menu').find('li');

			$sorting_buttons.on('click', function () {
				if ($(this).hasClass('active')) return false;
				$(this).parent().find('.active').removeClass('active');
				$(this).addClass('active');
				var filterValue = $(this).data('filter');
				if (typeof filterValue != "undefined") {
					$current.isotope({filter: filterValue});
					return false;
				}
			});
		});
	};

	// Ion.RangeSlider
	// version 2.2.0 Build: 380
	// © Denis Ineshin, 2017
	// https://github.com/IonDen
	//
	// Project page:    http://ionden.com/a/plugins/ion.rangeSlider/en.html
	// GitHub page:     https://github.com/IonDen/ion.rangeSlider
	//
	// Released under MIT licence:
	// http://ionden.com/a/plugins/licence-en.html

	CRUMINA.rangeSlider = function () {
		$(".range-slider-js").ionRangeSlider({
				type: "double",
				grid: true,
				min: 0,
				max: 1000,
				from: 200,
				to: 800,
				prefix: "$"
			}
		);
	};


	/* -----------------------------
	 * Sliders and Carousels
	 * ---------------------------*/

	CRUMINA.initSwiper = function () {
		var initIterator = 0;
		var $breakPoints = false;
		$('.swiper-container').each(function () {

			var $t = $(this);
			var index = 'swiper-unique-id-' + initIterator;

			$t.addClass('swiper-' + index + ' initialized').attr('id', index);
			$t.find('.swiper-pagination').addClass('pagination-' + index);

			var $effect = ($t.data('effect')) ? $t.data('effect') : 'slide',
				$crossfade = ($t.data('crossfade')) ? $t.data('crossfade') : true,
				$loop = ($t.data('loop') == false) ? $t.data('loop') : true,
				$showItems = ($t.data('show-items')) ? $t.data('show-items') : 1,
				$scrollItems = ($t.data('scroll-items')) ? $t.data('scroll-items') : 1,
				$scrollDirection = ($t.data('direction')) ? $t.data('direction') : 'horizontal',
				$mouseScroll = ($t.data('mouse-scroll')) ? $t.data('mouse-scroll') : false,
				$autoplay = ($t.data('autoplay')) ? parseInt($t.data('autoplay'), 10) : 0,
				$autoheight = ($t.hasClass('auto-height')) ? true : false,
				$slidesSpace = ($showItems > 1) ? 20 : 0;

			if ($showItems > 1) {
				$breakPoints = {
					480: {
						slidesPerView: 1,
						slidesPerGroup: 1
					},
					768: {
						slidesPerView: 2,
						slidesPerGroup: 2
					}
				}
			}

			swipers['swiper-' + index] = new Swiper('.swiper-' + index, {
				pagination: '.pagination-' + index,
				paginationClickable: true,
				direction: $scrollDirection,
				mousewheelControl: $mouseScroll,
				mousewheelReleaseOnEdges: $mouseScroll,
				slidesPerView: $showItems,
				slidesPerGroup: $scrollItems,
				spaceBetween: $slidesSpace,
				keyboardControl: true,
				setWrapperSize: true,
				preloadImages: true,
				updateOnImagesReady: true,
				autoplay: $autoplay,
				autoHeight: $autoheight,
				loop: $loop,
				breakpoints: $breakPoints,
				effect: $effect,
				fade: {
					crossFade: $crossfade
				},
				parallax: true,
				onImagesReady: function (swiper) {

				},
				onSlideChangeStart: function (swiper) {
					if ($t.find('.slider-slides').length) {
						$t.find('.slider-slides .slide-active').removeClass('slide-active');
						var realIndex = swiper.slides.eq(swiper.activeIndex).attr('data-swiper-slide-index');
						$t.find('.slider-slides .slides-item').eq(realIndex).addClass('slide-active');
					}
				}
			});
			initIterator++;
		});

		//swiper arrows
		$('.btn-prev').on('click', function () {
			swipers['swiper-' + $(this).parent().attr('id')].slidePrev();
		});

		$('.btn-next').on('click', function () {
			swipers['swiper-' + $(this).parent().attr('id')].slideNext();
		});

		//swiper tabs
		$('.slider-slides .slides-item').on('click', function () {
			if ($(this).hasClass('slide-active')) return false;
			var activeIndex = $(this).parent().find('.slides-item').index(this);
			swipers['swiper-' + $(this).closest('.swiper-container').attr('id')].slideTo(activeIndex + 1);
			$(this).parent().find('.slide-active').removeClass('slide-active');
			$(this).addClass('slide-active');

			return false;

		});
	};


	CRUMINA.burgerAnimation = function () {
		/* In animations (to close icon) */

		var beginAC = 80,
			endAC = 320,
			beginB = 80,
			endB = 320;

		function inAC(s) {
			s.draw('80% - 240', '80%', 0.3, {
				delay: 0.1,
				callback: function () {
					inAC2(s)
				}
			});
		}

		function inAC2(s) {
			s.draw('100% - 545', '100% - 305', 0.6, {
				easing: ease.ease('elastic-out', 1, 0.3)
			});
		}

		function inB(s) {
			s.draw(beginB - 60, endB + 60, 0.1, {
				callback: function () {
					inB2(s)
				}
			});
		}

		function inB2(s) {
			s.draw(beginB + 120, endB - 120, 0.3, {
				easing: ease.ease('bounce-out', 1, 0.3)
			});
		}

		/* Out animations (to burger icon) */

		function outAC(s) {
			s.draw('90% - 240', '90%', 0.1, {
				easing: ease.ease('elastic-in', 1, 0.3),
				callback: function () {
					outAC2(s)
				}
			});
		}

		function outAC2(s) {
			s.draw('20% - 240', '20%', 0.3, {
				callback: function () {
					outAC3(s)
				}
			});
		}

		function outAC3(s) {
			s.draw(beginAC, endAC, 0.7, {
				easing: ease.ease('elastic-out', 1, 0.3)
			});
		}

		function outB(s) {
			s.draw(beginB, endB, 0.7, {
				delay: 0.1,
				easing: ease.ease('elastic-out', 2, 0.4)
			});
		}

		/* Scale functions */

		function addScale(m) {
			m.className = 'menu-icon-wrapper scaled';
		}

		function removeScale(m) {
			m.className = 'menu-icon-wrapper';
		}

		/* Awesome burger scaled */

		var pathD = document.getElementById('pathD'),
			pathE = document.getElementById('pathE'),
			pathF = document.getElementById('pathF'),
			segmentD = new Segment(pathD, beginAC, endAC),
			segmentE = new Segment(pathE, beginB, endB),
			segmentF = new Segment(pathF, beginAC, endAC),
			wrapper2 = document.getElementById('menu-icon-wrapper'),
			trigger2 = document.getElementById('menu-icon-trigger'),
			toCloseIcon2 = true;

		wrapper2.style.visibility = 'visible';

		trigger2.onclick = function () {
			addScale(wrapper2);
			if (toCloseIcon2) {
				inAC(segmentD);
				inB(segmentE);
				inAC(segmentF);
			} else {
				outAC(segmentD);
				outB(segmentE);
				outAC(segmentF);

			}
			toCloseIcon2 = !toCloseIcon2;
			setTimeout(function () {
				removeScale(wrapper2)
			}, 450);
		};
	};

	/* -----------------------------
	 * On Click Functions
	 * ---------------------------*/


	$window.keydown(function (eventObject) {
		if (eventObject.which == 27) {
			if ($asidePanel.hasClass('opened')) {
				CRUMINA.togglePanel();
			}
			if ($body.hasClass('open')) {
				CRUMINA.toggleSearch();
			}
		}
	});

	jQuery(".js-window-popup").on('click', function () {
		setTimeout(function() {
			$('.window-popup').addClass('open');
			$body.toggleClass('body-overflow');
			}, 300);
		return false;
	});

	jQuery(".js-popup-close").on('click', function () {
		{
			$('.window-popup').removeClass('open');
			$body.removeClass('body-overflow')
		}
		return false;
	});

	jQuery(".js-close-aside").on('click', function () {
		if ($asidePanel.hasClass('opened')) {
			CRUMINA.togglePanel();
		}
		return false;
	});

	jQuery(".js-open-aside").on('click', function () {
		if (!$asidePanel.hasClass('opened')) {
			CRUMINA.togglePanel();
		}
		return false;
	});
	jQuery(".js-open-search").on('click', function () {
		CRUMINA.toggleSearch();
		return false;
	});

	jQuery(".overlay_search-close").on('click', function () {
		$body.removeClass('open');
		return false;
	});

	jQuery(".js-open-p-search").on('click', function () {
		$popupSearch.fadeToggle();
	});

	jQuery("#top-bar-js").on('click', function () {
		$('.top-bar').addClass('open');
		$body.toggleClass('overlay-enable');
		return false;
	});

	jQuery("#top-bar-close-js").on('click', function () {
		$('.top-bar').removeClass('open');
		$body.removeClass('overlay-enable');
		return false;
	});


	if ($popupSearch.length) {
		$popupSearch.find('input').focus(function () {
			$popupSearch.stop().animate({
				'width': $popupSearch.closest('.container').width() + 70
			}, 600)
		}).blur(function () {
			$popupSearch.fadeToggle('fast', function () {
				$popupSearch.css({
					'width': ''
				});
			});

		});
	}

	// Hide cart on click outside.
	$document.on('click', function (event) {
		if (!$(event.target).closest($cartPopap).length) {
			if ($cartPopap.hasClass('visible')) {
				$cartPopap.fadeToggle(200);
				$cartPopap.toggleClass('visible')
			}
		}
		if (!$(event.target).closest($asidePanel).length) {
			if ($asidePanel.hasClass('opened')) {
				CRUMINA.togglePanel();
			}
		}

	});

	// Show dropdown cart on icon click.
	jQuery(".js-cart-animate").on('click', function (event) {
		event.stopPropagation();
		$cartPopap.toggleClass('visible');
		$cartPopap.fadeToggle(200);
	});


	$('.quantity-plus').on('click', function () {
		var val = parseInt($(this).prev('input').val());
		$(this).prev('input').val(val + 1).change();
		return false;
	});

	$('.quantity-minus').on('click', function () {
		var val = parseInt($(this).next('input').val());
		if (val !== 1) {
			$(this).next('input').val(val - 1).change();
		}
		return false;
	});

	/*---------------------------------
	 ACCORDION
	 -----------------------------------*/
	jQuery('.accordion-heading').on('click', function () {
		jQuery(this).parents('.panel-heading').toggleClass('active');
		jQuery(this).parents('.accordion-panel').toggleClass('active');
	});

	//Scroll to top.
	jQuery('.back-to-top').on('click', function () {
		$('html,body').animate({
			scrollTop: 0
		}, 1200);
		return false;
	});

	jQuery(".input-inline").find('input').focus(function () {
		$(this).closest('form').addClass('input-drop-shadow');
	}).blur(function () {
		$(this).closest('form').removeClass('input-drop-shadow');
	});

	/* -----------------------
* Create the map
* https://leafletjs.com/
* --------------------- */

	CRUMINA.maps = {
		maps: {
			mapUSA: {
				config: {
					id: 'map',
					map: {
						center: new L.LatLng(38.897663, -77.036575),
						zoom: 12,
						maxZoom: 18,
						layers: new L.tileLayer('https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png', {
							maxZoom: 18,
							attribution: '',
						})
					},
					icon: {
						iconSize: [56, 73],
						iconAnchor: [22, 94],
						className: 'icon-map'
					}
				},
				markers: [
					{
						coords: [38.897663, -77.036575],
						icon: 'marker-google.png'
					}
				]
			}
		},
		init: function () {
			var _this = this;

			for (var key in this.maps) {
				var data = this.maps[key];

				if (!data.config || !data.markers) {
					continue;
				}

				if (!document.getElementById(data.config.id)) {
					continue;
				}

				var map = new L.map(data.config.id, data.config.map);
				var cluster = L.markerClusterGroup({
					iconCreateFunction: function (cluster) {
						var childCount = cluster.getChildCount();
						var config = data.config.cluster;
						return new L.DivIcon({
							html: '<div><span>' + childCount + '</span></div>',
							className: 'marker-cluster marker-cluster-' + key,
							iconSize: new L.Point(config.iconSize[0], config.iconSize[1])
						});
					}
				});
				data.markers.forEach(function (item) {
					data.config.icon['iconUrl'] = './img/' + item.icon;
					var icon = L.icon(data.config.icon);

					var marker = L.marker(item.coords, {icon: icon});
					cluster.addLayer(marker);
				});

				map.addLayer(cluster);
				this.disableScroll(jQuery("#" + data.config.id), map);
			}
		},
		disableScroll: function ($map, map) {
			map.scrollWheelZoom.disable();

			$map.bind('mousewheel DOMMouseScroll', function (event) {
				event.stopPropagation();
				if (event.ctrlKey == true) {
					event.preventDefault();
					map.scrollWheelZoom.enable();
					setTimeout(function () {
						map.scrollWheelZoom.disable();
					}, 1000);
				} else {
					map.scrollWheelZoom.disable();
				}
			});
		}
	};


	/* -----------------------------
	 * On DOM ready functions
	 * ---------------------------*/

	$document.ready(function () {

		if ($('#menu-icon-wrapper').length) {
			CRUMINA.burgerAnimation();
		}
		// 3-d party libs run
		$primaryMenu.crumegamenu({
			showSpeed: 0,
			hideSpeed: 0,
			trigger: "hover",
			animation: "drop-up",
			indicatorFirstLevel: "&#xf0d7",
			indicatorSecondLevel: "&#xf105"
		});

		CRUMINA.fixedHeader();
		CRUMINA.initSwiper();
		CRUMINA.equalHeight();
		CRUMINA.customScroll();
		CRUMINA.mediaPopups();
		CRUMINA.IsotopeSort();
		CRUMINA.parallaxFooter();
		CRUMINA.rangeSlider();
		CRUMINA.preloader();
		CRUMINA.headerSpacer.init();
		CRUMINA.updateResponsiveInit();
		CRUMINA.maps.init();

		// Dom mofifications
		$('select').niceSelect();
		FastClick.attach(document.body);


		// Smooth scroll

		if (typeof SmoothScroll === "function") {
			CRUMINA.smoothScroll = new SmoothScroll('a[href*="#"]', {
				header: '#site-header'
			});
		}


		// On Scroll animations.
		CRUMINA.animateSvg();
		CRUMINA.counters();
		CRUMINA.progresBars();
		CRUMINA.pieCharts();

		// Row background animation
		if ($('.subscribe').length) {
			CRUMINA.SubscribeScrollAnnimation();
		}
		if ($('.seo-score').length) {
			CRUMINA.SeoScoreScrollAnnimation();
		}
		if ($('.testimonial-slider').length) {
			CRUMINA.TestimonialScrollAnnimation();
		}
		if ($('.our-vision').length) {
			CRUMINA.OurVisionScrollAnnimation();
		}
		if ($('.background-mountains').length) {
			CRUMINA.MountainsScrollAnnimation();
		}
	});
})(jQuery);
