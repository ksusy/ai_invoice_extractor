"""SQLAlchemy 2.0 ORM models for the invoice processing platform.

This module defines the complete relational database schema including:
- Core invoice table with Czech-to-English field translations
- Commodity-specific detail tables (electricity NN/VN, gas, water)
- Analytics and processing tracking tables

All field names follow English conventions with Czech mappings documented
in docs/domain_mapping_cz_en.md.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID_TYPE
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# V produkci běží PostgreSQL a využívají se jeho nativní typy JSONB a UUID.
# Testy běží nad SQLite, který je nezná — varianta typu proto v tomto případě
# spadne zpět na obecné JSON a Uuid, aniž by se měnilo produkční schéma.
JSONB = PG_JSONB().with_variant(JSON(), "sqlite")
PG_UUID = PG_UUID_TYPE(as_uuid=True).with_variant(Uuid(as_uuid=True), "sqlite")

if TYPE_CHECKING:
    pass


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


# ════════════════════════════════════════════════════════════════════════════
# CORE INVOICE TABLE (doklady / faktury)
# ════════════════════════════════════════════════════════════════════════════


class Invoice(Base):
    """Core invoice record – main document log.

    Czech field mapping:
        - id_doklad -> id
        - kod_odberne_misto -> supply_point_code
        - doklad_cislo -> invoice_number
        - obdobi_od -> period_from
        - obdobi_do -> period_to
        - ico_odberatel -> customer_cin
        - ico_dodavatel -> supplier_cin
        - opravna -> is_correction
        - kterou_fakturu_opravuje -> corrected_invoice_id
        - datum_vystaveni -> issue_date
        - datum_splatnosti -> due_date
        - prechodova_faktura -> is_transitional
    """

    __tablename__ = "invoices"

    # Primary key (id_doklad)
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )

    # Source file tracking
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    transaction_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("transactions.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Invoice identification (doklad_cislo)
    invoice_number: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Supply point (kod_odberne_misto / EAN / EIC merged)
    supply_point_code: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Commodity type
    commodity: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # elektrina_nn | elektrina_vn | plyn_MO | plyn_VO | teplo | voda

    # Billing period (obdobi_od / obdobi_do)
    period_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Key dates
    issue_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # datum_vystaveni
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # datum_splatnosti
    tax_point_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # datum_uzp / DUZP

    # Party identification (IČO) — stored as zero-padded 8-char string to preserve leading zeros.
    # e.g. IČO "01234567" must not be stored as integer 1234567.
    customer_cin: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)  # ico_odberatel
    supplier_cin: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)  # ico_dodavatele

    # Monetary amounts (all in CZK)
    total_amount_ex_vat: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )  # castka_bez_dph
    total_amount_inc_vat: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )  # castka_s_dph
    vat_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # dan_zakladni
    vat_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)  # sazba_dph (%)

    # Correction invoice flags (opravná faktura)
    is_correction: Mapped[bool] = mapped_column(Boolean, default=False)  # opravna
    corrected_invoice_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("invoices.id", ondelete="SET NULL"), nullable=True
    )  # kterou_fakturu_opravuje

    # Transitional invoice flag (přechodová faktura)
    is_transitional: Mapped[bool] = mapped_column(Boolean, default=False)  # prechodova_faktura

    # Processing metadata
    extraction_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    # Full extracted data as JSON backup
    raw_extracted_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Relationships ────────────────────────────────────────────
    corrected_invoice: Mapped[Invoice | None] = relationship(
        "Invoice", remote_side=[id], foreign_keys=[corrected_invoice_id]
    )
    electricity_nn_details: Mapped[list[ElectricityNNDetail]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )
    electricity_vn_details: Mapped[list[ElectricityVNDetail]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )
    gas_mo_details: Mapped[list[GasMODetail]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )
    water_details: Mapped[list[WaterDetail]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )
    heat_details: Mapped[list[HeatDetail]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )
    gas_vo_details: Mapped[list[GasVODetail]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )


# ════════════════════════════════════════════════════════════════════════════
# COMMODITY DETAIL TABLES (One-to-Many for transitional invoice support)
# ════════════════════════════════════════════════════════════════════════════


class ElectricityNNDetail(Base):
    """Electricity low voltage (elektřina nízké napětí) detail record.

    Czech field mapping:
        - spotreba_nt -> consumption_low_tariff (nízký tarif)
        - spotreba_vt -> consumption_high_tariff (vysoký tarif)
        - castka_bez_dph -> amount_ex_vat
        - castka_s_dph -> amount_inc_vat
        - jistic -> circuit_breaker_value
        - distribucni_tarif -> distribution_tariff
    """

    __tablename__ = "electricity_nn_details"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Period for this detail (allows split periods for transitional invoices)
    period_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Consumption (spotřeba)
    consumption_low_tariff: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 3), nullable=True
    )  # spotreba_nt (kWh)
    consumption_high_tariff: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 3), nullable=True
    )  # spotreba_vt (kWh)
    total_consumption: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 3), nullable=True
    )  # celková spotřeba (kWh)

    # Meter readings (stavy měřidla)
    meter_reading_start: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    meter_reading_end: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)

    # Tariff and technical info
    distribution_tariff: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # distribucni_tarif (D01d, D02d, etc.)
    circuit_breaker_value: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 2), nullable=True
    )  # jistic (A)

    # Amounts (částky)
    amount_ex_vat: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )  # castka_bez_dph
    amount_inc_vat: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )  # castka_s_dph

    # Component breakdown (optional)
    supply_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # silová elektřina
    distribution_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # distribuce
    system_services: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # systémové služby
    renewable_energy_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # POZE

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    invoice: Mapped[Invoice] = relationship(back_populates="electricity_nn_details")


class ElectricityVNDetail(Base):
    """Electricity high voltage (elektřina vysoké napětí) detail record."""

    __tablename__ = "electricity_vn_details"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Period for this detail
    period_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Consumption (MWh)
    supply_consumption: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # spotreba_se
    peak_consumption: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    off_peak_consumption: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)

    # Supply charges
    supply_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # castka_se
    supply_tax_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # castka_dan_se

    # Quarter-hour maximum
    quarter_hour_max: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)  # ctvrt_hod_max
    eru_rate: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # sazba_eru

    # Reserved capacity (MW)
    annual_reserved_capacity: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)  # rk_rocni
    annual_reserved_capacity_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # castka_rk_rocni
    monthly_reserved_capacity: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)  # rk_mesicni
    monthly_reserved_capacity_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # castka_rk_mesicni

    # Grid usage
    grid_usage_rate: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # sazba_pouziti_siti
    grid_usage_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # castka_pouziti_siti

    # Capacity excess
    reserved_capacity_excess: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # prekroceni_rk
    reserved_capacity_excess_rate: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # sazba_prekroceni_rk
    reserved_capacity_excess_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # castka_prekroceni_rk

    # Reactive power (kVArh)
    power_factor: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)  # tg_fi_vn
    reactive_power_quantity: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # mnozstvi_jalova
    reactive_power_rate: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # sazba_jalova
    reactive_power_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # castka_jalova

    # Fees / services
    service_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_sluzby
    operating_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_provoz
    renewable_energy_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # poze

    # Amounts (CZK)
    amount_ex_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    amount_inc_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    invoice: Mapped[Invoice] = relationship(back_populates="electricity_vn_details")


class GasMODetail(Base):
    """Gas small-scale consumer (plyn maloodběr) detail record.

    Czech field mapping:
        - koef_prepoctu -> conversion_factor (přepočtový koeficient)
        - spalne_teplo -> combustion_heat (spalné teplo MJ/m³)
        - spotreba_mwh -> consumption_mwh
        - spotreba_m3 -> consumption_m3
    """

    __tablename__ = "gas_mo_details"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Period for this detail
    period_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Consumption (spotřeba)
    consumption_m3: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 3), nullable=True
    )  # spotreba_m3
    consumption_mwh: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 6), nullable=True
    )  # spotreba_mwh

    # Conversion parameters
    conversion_factor: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6), nullable=True
    )  # koef_prepoctu
    combustion_heat: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True
    )  # spalne_teplo (MJ/m³)

    # Meter readings
    meter_reading_start: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    meter_reading_end: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)

    # Amounts
    amount_ex_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    amount_inc_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    # Component breakdown (legacy names)
    commodity_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # komodita (legacy alias)
    distribution_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # distribuce (legacy alias)
    fixed_monthly_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_za_staly_mesicni_plat

    # Full price breakdown
    period_months: Mapped[int | None] = mapped_column(Integer, nullable=True)  # obdobi_mesice
    commodity_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)  # jedn_cena_komoditni_slozky
    commodity_total_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_za_komoditni_slozku_ceny
    fixed_monthly_fee_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)  # jedn_cena_staly_mesicni_plat
    distribution_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)  # jedn_cena_distr_plynu
    distribution_fixed_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # pevna_cena_distribuce
    reserved_capacity_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)  # jedn_cena_pristavena_kapacita
    reserved_capacity_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_za_pristavenou_kapacitu
    market_operator_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_za_cinnost_operatora_trhu
    natural_gas_tax_total: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # dan_zemni_plyn_celkem

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    invoice: Mapped[Invoice] = relationship(back_populates="gas_mo_details")


class WaterDetail(Base):
    """Water supply and sewage (voda) detail record."""

    __tablename__ = "water_details"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Period for this detail
    period_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Consumption (m³)
    consumption_m3: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)

    # DEPRECATED: meter readings
    meter_reading_start: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    meter_reading_end: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)

    # Rate components (CZK)
    water_rate: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # vodne
    sewage_rate: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # stocne
    precipitation_water: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # srazkove
    wastewater_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # odpadni

    # Amounts (CZK)
    amount_ex_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    amount_inc_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    invoice: Mapped[Invoice] = relationship(back_populates="water_details")


class HeatDetail(Base):
    """District heating (teplo) detail record."""

    __tablename__ = "heat_details"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Period for this detail
    period_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Consumption
    consumption_gj: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # spotreba_gj (legacy)
    heat_consumption: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # spotreba_tepla
    hot_water_heating: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # spotreba_ohrev_tv
    cold_water: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # studena_voda
    total_heat_consumption: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # celkova_spotreba_tepla

    # Capacity / water
    reserved_capacity: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # rez_kapacita
    supplementary_water: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # doplnovaci_voda

    # DEPRECATED: property info
    heated_area: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)  # vytapena_plocha

    # Charges (CZK)
    fixed_monthly_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # staly_mesicni_plat
    variable_charge: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    # Amounts (CZK)
    amount_ex_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    amount_inc_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    invoice: Mapped[Invoice] = relationship(back_populates="heat_details")


class GasVODetail(Base):
    """Gas large-scale consumer (plyn velkoodběr) detail record."""

    __tablename__ = "gas_vo_details"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Period for this detail
    period_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Consumption
    consumption_m3: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # spotreba_m
    consumption_mwh: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # spotreba_mwh

    # Conversion parameters
    conversion_factor: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)  # koef_prepoctu
    combustion_heat: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)  # spalne_teplo

    # Capacity
    daily_reserved_capacity: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)  # mnozstvi_denni_rez_kapacity

    # Commodity / trade pricing
    other_supply_services_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_ostatni_sluzby_dodavky
    trade_reserved_capacity_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # jednotkova_cena_za_obchod_rez_kapacity
    trade_reserved_capacity_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_za_obchod_rez_kapacity

    # Distribution
    distribution_service_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_sluzby_distribuce
    distribution_system_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # jednotkova_cena_za_sluzby_distr_soustavy
    distribution_reserved_capacity_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)  # jednotkova_cena_za_distribuci_rez_kapacity
    distribution_reserved_capacity_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_distribuce_rezervovane_kapacity

    # Other fees
    market_operator_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # cena_cinnost_operatora_trhu
    natural_gas_tax_total: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)  # dan_zemni_plyn_celkem

    # Amounts (CZK)
    amount_ex_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    amount_inc_vat: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    invoice: Mapped[Invoice] = relationship(back_populates="gas_vo_details")


# ════════════════════════════════════════════════════════════════════════════
# ANALYTICS & PROCESSING TABLES
# ════════════════════════════════════════════════════════════════════════════


class Transaction(Base):
    """High-level record of an invoice processing transaction."""

    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)  # Path in data/raw/
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    commodity: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Classification result
    is_scan: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )  # True = scanned image, False = native PDF with text layer, None = not classified yet

    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True
    )  # pending | classifying | processing | completed | error
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    ocr_results: Mapped[list[OCRResult]] = relationship(
        back_populates="transaction", cascade="all, delete-orphan"
    )
    extraction_results: Mapped[list[DBExtractionResult]] = relationship(
        back_populates="transaction", cascade="all, delete-orphan"
    )


class OCRResult(Base):
    """Stores raw OCR output and metrics for each engine run."""

    __tablename__ = "ocr_results"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    engine_name: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # tesseract | paddleocr | easyocr
    raw_text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    transaction: Mapped[Transaction] = relationship(back_populates="ocr_results")


class DBExtractionResult(Base):
    """Stores structured extraction output, user corrections, and accuracy metrics."""

    __tablename__ = "extraction_results"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ocr_result_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("ocr_results.id", ondelete="SET NULL"), nullable=True, index=True
    )
    strategy_name: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )  # regex | llm_text | vision_llm | hybrid
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    extracted_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    user_corrected_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    fields_corrected: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    transaction: Mapped[Transaction] = relationship(back_populates="extraction_results")


class PipelineStepLog(Base):
    """Log entry for each step in the processing pipeline."""

    __tablename__ = "pipeline_step_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID, primary_key=True, default=uuid.uuid4
    )
    invoice_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="started", index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
