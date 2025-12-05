import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ForecastAqi } from './forecast-aqi';

describe('ForecastAqi', () => {
  let component: ForecastAqi;
  let fixture: ComponentFixture<ForecastAqi>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ForecastAqi]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ForecastAqi);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
