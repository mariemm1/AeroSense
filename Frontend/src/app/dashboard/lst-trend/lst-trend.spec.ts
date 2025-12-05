import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LstTrend } from './lst-trend';

describe('LstTrend', () => {
  let component: LstTrend;
  let fixture: ComponentFixture<LstTrend>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LstTrend]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LstTrend);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
