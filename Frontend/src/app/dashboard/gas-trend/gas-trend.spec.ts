import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GasTrend } from './gas-trend';

describe('GasTrend', () => {
  let component: GasTrend;
  let fixture: ComponentFixture<GasTrend>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GasTrend]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GasTrend);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
