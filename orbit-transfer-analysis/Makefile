# 프로젝트 루트 Makefile — 영문/한국어 논문 빌드 통합
EN_DIR   = manuscript
KO_DIR   = manuscript-ko

.PHONY: all en ko clean clean-en clean-ko view-en view-ko \
        quick quick-ko bib bib-ko distclean watch watch-ko

# 기본 타겟: 영문 빌드 후 열기
all:
	$(MAKE) -C $(EN_DIR) pdf
	$(MAKE) -C $(EN_DIR) view

# 한국어: 빌드 후 열기
ko:
	$(MAKE) -C $(KO_DIR) pdf
	$(MAKE) -C $(KO_DIR) view

# ── 영문 보조 타겟 ───────────────────────────
view-en:
	$(MAKE) -C $(EN_DIR) view

quick:
	$(MAKE) -C $(EN_DIR) quick

bib:
	$(MAKE) -C $(EN_DIR) bib

watch:
	$(MAKE) -C $(EN_DIR) watch

clean-en:
	$(MAKE) -C $(EN_DIR) clean

# ── 한국어 보조 타겟 ─────────────────────────
view-ko:
	$(MAKE) -C $(KO_DIR) view

quick-ko:
	$(MAKE) -C $(KO_DIR) quick

bib-ko:
	$(MAKE) -C $(KO_DIR) bib

watch-ko:
	$(MAKE) -C $(KO_DIR) watch

clean-ko:
	$(MAKE) -C $(KO_DIR) clean

# ── 공통 ─────────────────────────────────────
clean: clean-en clean-ko

distclean:
	$(MAKE) -C $(EN_DIR) distclean
	$(MAKE) -C $(KO_DIR) distclean
